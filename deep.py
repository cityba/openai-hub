import sys
import json
import time
import os
import psutil
import requests
import logging
import platform
from typing import List, Dict
import re


from cryptography.fernet import Fernet


from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QTabWidget, QCheckBox, QSplitter,
    QMessageBox, QToolBar, QAction, QStatusBar, QFileDialog, QMenu, QToolButton,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QSizePolicy,
    QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QSize, QTimer, QObject
from PyQt5.QtGui import (
    QTextCursor, QPalette, QColor, QFont, QIcon, QTextCharFormat,
    QKeySequence
)


import gc
import tracemalloc
import asyncio
import aiohttp


# Próbáljuk importálni a QScintilla-t
try:
    from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciLexerCPP, QsciLexerJava, QsciLexerJavaScript


    HAS_SCINTILLA = True
except ImportError:
    HAS_SCINTILLA = False


# Rendszeroptimalizációk
try:
    process = psutil.Process(os.getpid())
    if platform.system() == 'Windows':
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    else:
        process.nice(-18)
except Exception as e:
    logging.error(f"Prioritás beállítási hiba: {str(e)}")


# Környezeti változók
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)


# Alkalmazás konstansok
APP_NAME = "SzitaAIPro"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_URL = "https://openrouter.ai/api/v1/models"


MAX_HISTORY = 15
MAX_FILE_SIZE = 30000
TOKEN_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072]



class SettingsManager:
    def __init__(self):
        self.config_dir = os.path.join(os.getenv('APPDATA'), APP_NAME)
        os.makedirs(self.config_dir, exist_ok=True)
        self.settings = QSettings(os.path.join(self.config_dir, 'config.ini'), QSettings.IniFormat)


    def get(self, key: str, default=None):
        return self.settings.value(key, default)


    def set(self, key: str, value):
        self.settings.setValue(key, value)


    @property
    def history_dir(self):
        path = os.path.join(self.config_dir, 'history')
        os.makedirs(path, exist_ok=True)
        return path



settings = SettingsManager()



class EncryptionManager:
    def __init__(self):
        key = settings.get('encryption_key')
        if not key:
            key = Fernet.generate_key().decode()
            settings.set('encryption_key', key)
        self.cipher = Fernet(key.encode())


    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()


    def decrypt(self, data: str) -> str:
        return self.cipher.decrypt(data.encode()).decode()



encryptor = EncryptionManager()



class NetworkManager(QThread):
    models_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)


    def __init__(self):
        super().__init__()
        self.free_only = True


    async def fetch_models(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(MODEL_URL, timeout=6) as response:
                    response.raise_for_status()
                    data = await response.json()
                    parsed = self.parse_models(data.get('data', []))
                    self.models_loaded.emit(parsed)
        except aiohttp.ClientError as e:
            self.error_occurred.emit(f"Hálózati hiba: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Hálózati hiba: {str(e)}")


    def run(self):
        asyncio.run(self.fetch_models())


    def parse_models(self, models: List[Dict]) -> List[str]:
        result = []
        for model in models:
            model_id = model.get('id', '')
            if not any(provider in model_id for provider in
                       ['deepseek', 'openrouter', 'google', 'mistral', 'meta', 'moonshotai', 'anthropic']):
                continue


            context = model.get('context_length')
            if not isinstance(context, int) or context < 64000:
                continue


            pricing = model.get('pricing', {})
            is_free = pricing.get('prompt') == "0" and pricing.get('completion') == "0"


            if self.free_only and not is_free:
                continue


            tokens = context // 1024 if context else 0
            label = f"{model_id} | {tokens}K {' 🆓 ' if is_free else '💲'}"
            result.append(label)


        return result



class AIWorker(QThread):
    update_received = pyqtSignal(str)
    response_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, int)
    truncated = pyqtSignal()


    def __init__(self, api_key: str, messages: List[Dict], model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model = model.split('|')[0].strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.running = True
        self.session = requests.Session()


    def run(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": {"exclude": True},
            "transforms": ["middle-out"],
            "usage": {"include": True},
            "stream": True
        }


        try:
            response = self.session.post(
                API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )


            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error = error_data.get('error', {}).get('message', 'Ismeretlen hiba')
                except:
                    error = response.text[:200] + "..." if len(response.text) > 200 else response.text
                self.error_occurred.emit(error, response.status_code)
                return


            buffer = bytearray()  # Use bytearray for efficient binary handling
            for chunk in response.iter_lines():
                if not self.running:
                    return


                if chunk:
                    try:
                        # Decode directly to bytes
                        decoded = chunk
                        if not decoded:
                            continue


                        if decoded.startswith(b'data:'):
                            data = decoded[5:].strip()
                        else:
                            data = decoded.strip()


                        if not data:
                            continue


                        if not data.startswith(b'{'):
                            logging.debug(f"Non-JSON response: {data}")
                            continue


                        try:
                            parsed = json.loads(data.decode('utf-8'))  # Decode for JSON parsing
                        except json.JSONDecodeError as e:
                            logging.error(f"JSONDecodeError: {e}, Data: {data}")
                            continue


                        choices = parsed.get('choices', [{}])
                        if choices:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                buffer.extend(content.encode('utf-8'))  # Encode to bytes
                                self.update_received.emit(buffer.decode('utf-8', errors='replace'))
                                buffer.clear()  # Clear the buffer
                            else:
                                self.update_received.emit(content)
                                buffer.clear()  # Clear the buffer


                            finish_reason = choices[0].get('finish_reason')
                            if finish_reason == 'length':
                                self.truncated.emit()


                    except UnicodeDecodeError as e:
                        logging.error(f"UnicodeDecodeError: {e}")
                    except Exception as e:
                        logging.error(f"Váratlan hiba: {str(e)}")


            if buffer:
                self.update_received.emit(buffer.decode('utf-8', errors='replace'))


            self.response_completed.emit("Kész!")


        except requests.RequestException as e:
            self.error_occurred.emit(f"Hálózati hiba: {str(e)}", 500)
        except Exception as e:
            self.error_occurred.emit(f"Váratlan hiba: {str(e)}", 500)


    def stop(self):
        self.running = False



class CodeEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)


        if HAS_SCINTILLA:
            self.editor = QsciScintilla()
            self.editor.setAutoIndent(True)
            self.editor.setIndentationGuides(True)
            self.editor.setMarginLineNumbers(1, True)
            self.editor.setMarginWidth(1, "0000")
            self.editor.setBraceMatching(QsciScintilla.SloppyBraceMatch)
            self.editor.setCaretLineVisible(True)
            self.editor.setCaretLineBackgroundColor(QColor(30, 30, 40))
        else:
            self.editor = QPlainTextEdit()
            self.editor.setReadOnly(True)


        self.layout.addWidget(self.editor)


    def set_language(self, language):
        if not HAS_SCINTILLA:
            return


        lexer = None
        lang_map = {
            "python": QsciLexerPython,
            "cpp": QsciLexerCPP,
            "java": QsciLexerJava,
            "javascript": QsciLexerJavaScript
        }


        if language.lower() in lang_map:
            lexer = lang_map[language.lower()]()


        self.editor.setLexer(lexer)


    def setText(self, text):
        if HAS_SCINTILLA:
            self.editor.setText(text)
        else:
            self.editor.setPlainText(text)


    def text(self):
        if HAS_SCINTILLA:
            return self.editor.text()
        else:
            return self.editor.toPlainText()



class SearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keresés")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)


        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()


        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self.on_search_text_changed)
        self.form_layout.addRow("Keresés:", self.search_edit)


        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.find_next)
        self.buttons.rejected.connect(self.reject)
        self.layout.addLayout(self.form_layout)


        self.find_next_button = QPushButton("Következő")
        self.find_prev_button = QPushButton("Előző")
        self.buttons.addButton(self.find_next_button, QDialogButtonBox.ActionRole)
        self.buttons.addButton(self.find_prev_button, QDialogButtonBox.ActionRole)
        self.find_next_button.clicked.connect(self.find_next)
        self.find_prev_button.clicked.connect(self.find_prev)


        self.layout.addWidget(self.buttons)
        self.editor = parent.chat_display
        self.cursor = self.editor.textCursor()


    def on_search_text_changed(self, text):
        self.cursor = self.editor.textCursor()
        self.cursor.setPosition(0)


    def find_next(self):
        text_to_find = self.search_edit.text()
        if not text_to_find:
            return


        cursor = self.editor.document().find(text_to_find, self.cursor)
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)
            self.cursor = cursor
        else:
            QMessageBox.information(self, "Keresés", "Nincs több találat.")
            self.cursor.setPosition(0)
            self.find_next()


    def find_prev(self):
        text_to_find = self.search_edit.text()
        if not text_to_find:
            return


        cursor = self.editor.document().find(text_to_find, self.cursor, QTextDocument.FindBackward)
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)
            self.cursor = cursor
        else:
            QMessageBox.information(self, "Keresés", "Nincs több találat.")
            self.cursor.movePosition(QTextCursor.End)
            self.find_prev()



class TextReceiver(QObject):
    """
    Egy jelzőfogadó osztály, ami a szöveget a fő szálon fogadja.
    """
    update_text = pyqtSignal(str)



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        self.encryption_manager = EncryptionManager()
        self.network_manager = NetworkManager()
        self.worker = None
        self.history = []
        self.current_prompt = ""
        self.code_blocks = []
        self.buffered_text = ""
        self.update_interval = 100  # ms
        self.text_receiver = TextReceiver()  # Jelzőfogadó példányosítása
        self.text_receiver.update_text.connect(self.append_to_chat)  # Jelző összekötése
        self.is_generating = False
        self.code_tab_count = 0 # Számláló a kód tabokhoz


        self.setup_ui()
        self.setup_connections()
        self.load_settings()
        self.setWindowIcon(self.get_application_icon())
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.flush_buffer)


        tracemalloc.start()
        gc.collect()
        self.log_memory_usage()


    def log_memory_usage(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logging.info("Top 10 memóriahasználati hely:")
        for stat in top_stats[:10]:
            logging.info(stat)


    def handle_update(self, text: str):
        self.buffered_text += text
        if not self.update_timer.isActive():
            self.update_timer.start(self.update_interval)


    def flush_buffer(self):
        if self.buffered_text:
            self.text_receiver.update_text.emit(self.buffered_text)  # Jelzés küldése
            self.buffered_text = ""
        if self.worker and not self.worker.isRunning():
            self.update_timer.stop()


    def get_icon_path(self, icon_name):
        paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), icon_name),
            os.path.join(self.settings.config_dir, icon_name)
        ]
        if hasattr(sys, '_MEIPASS'):
            paths.append(os.path.join(sys._MEIPASS, icon_name))


        for path in paths:
            if os.path.exists(path):
                return path
        return None


    def get_icon(self, icon_name):
        icon_path = self.get_icon_path(icon_name)
        return QIcon(icon_path) if icon_path else QIcon()


    def get_application_icon(self):
        return self.get_icon("icon.ico") or self.get_icon("icon.png") or QIcon()


    def autosave_history(self):
        if not self.history:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.settings.history_dir, f"autosave_{timestamp}.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Automatikus mentés hiba: {str(e)}")


    def setup_ui(self):
        self.setWindowTitle("Szita AI Kódasszisztens")
        self.setMinimumSize(1000, 700)
        self.setWindowIcon(self.get_application_icon())


        main_layout = QVBoxLayout()
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)


        splitter = QSplitter(Qt.Horizontal)


        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)


        api_group = QGroupBox("API Beállítások")
        api_layout = QVBoxLayout(api_group)


        key_layout = QHBoxLayout()
        self.key_combo = QComboBox()
        self.key_combo.setEditable(True)
        self.key_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        key_layout.addWidget(QLabel("Kulcs:"))
        key_layout.addWidget(self.key_combo, 1)
        api_layout.addLayout(key_layout)


        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.free_check = QCheckBox("Ingyenes")
        self.free_check.setChecked(True)
        model_layout.addWidget(self.free_check)
        model_layout.addWidget(QLabel("Modell:"))
        model_layout.addWidget(self.model_combo, 1)
        api_layout.addLayout(model_layout)


        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Hőmérséklet:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.4)
        param_layout.addWidget(self.temp_spin)


        param_layout.addWidget(QLabel("Max tokenek:"))
        self.token_combo = QComboBox()
        for tokens in TOKEN_OPTIONS:
            self.token_combo.addItem(f"{tokens // 1024}K", tokens)
        self.token_combo.setCurrentIndex(3)
        param_layout.addWidget(self.token_combo)


        api_layout.addLayout(param_layout)
        left_layout.addWidget(api_group)


        input_group = QGroupBox("Kérés")
        input_layout = QVBoxLayout(input_group)
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Írd ide a kérdésed...")
        input_layout.addWidget(self.input_edit)


        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Fájl feltöltés")
        self.send_btn = QPushButton("Küldés")
        self.cont_btn = QPushButton("Folytatás")
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.cont_btn)
        input_layout.addLayout(btn_layout)
        left_layout.addWidget(input_group, 1)
        splitter.addWidget(left_panel)


        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)


        # Keresőmező hozzáadása
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Keresés a chatben...")
        self.search_bar.textChanged.connect(self.search_chat)
        right_layout.addWidget(self.search_bar)


        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)


        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.tab_widget.addTab(self.chat_display, "Chat")


        self.copy_btn = QPushButton("Kód másolása")
        self.copy_btn.clicked.connect(self.copy_code)
        self.copy_btn.setEnabled(False)


        right_layout.addWidget(self.tab_widget, 1)
        btn_layout_right = QHBoxLayout()
        btn_layout_right.addWidget(self.copy_btn)
        btn_layout_right.addStretch(1)
        right_layout.addLayout(btn_layout_right)


        self.status_bar = QStatusBar()
        right_layout.addWidget(self.status_bar)


        splitter.addWidget(self.right_panel)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)
        self.apply_dark_theme()


    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))


        save_action = QAction(QIcon.fromTheme('document-save'), "Mentés", self)
        save_action.triggered.connect(self.save_chat)
        toolbar.addAction(save_action)


        load_action = QAction(QIcon.fromTheme('document-open'), "Betöltés", self)
        load_action.triggered.connect(self.load_chat)
        toolbar.addAction(load_action)


        clear_action = QAction(QIcon.fromTheme('edit-clear'), "Törlés", self)
        clear_action.triggered.connect(self.clear_chat_display)
        toolbar.addAction(clear_action)


        clear_history_action = QAction(QIcon.fromTheme('edit-clear'), "Előzmények törlése", self)
        clear_history_action.triggered.connect(self.clear_history)
        toolbar.addAction(clear_history_action)


        toolbar.addSeparator()


        self.history_menu = QMenu("Előzmények", self)
        menu_button = QToolButton()
        menu_button.setText("Előzmények")
        menu_button.setMenu(self.history_menu)
        menu_button.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(menu_button)


        key_menu = QMenu("Kulcsok", self)
        add_action = QAction("Új kulcs hozzáadása", self)
        add_action.triggered.connect(self.add_api_key)
        key_menu.addAction(add_action)
        toolbar.addAction(key_menu.menuAction())


        self.update_history_menu()
        return toolbar


    def setup_connections(self):
        self.send_btn.clicked.connect(self.send_request)
        self.cont_btn.clicked.connect(self.continue_request)
        self.upload_btn.clicked.connect(self.upload_file)
        self.free_check.stateChanged.connect(self.refresh_models)
        self.tab_widget.currentChanged.connect(self.update_copy_button_state)


        self.refresh_models()
        self.load_api_keys()


        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.send_request)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self).activated.connect(self.continue_request)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self.show_search_dialog)


    def show_search_dialog(self):
        self.search_dialog = SearchDialog(self)
        self.search_dialog.show()


    def update_copy_button_state(self, index):
        self.copy_btn.setEnabled(index > 0)


    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.ToolTipBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipText, QColor(200, 200, 200))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
        palette.setColor(QPalette.HighlightedText, QColor(240, 240, 240))


        self.setPalette(palette)
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:disabled {
                 background-color: #5e615f;
                  color: #A0A0A0;
            }
            QStatusBar {
                color: #A0A0A0;
                font-size: 12px;
                background-color: #252526;
                border-top: 1px solid #3F3F46;
            }
            QTabWidget::pane {
                border: 1px solid #3F3F46;
                background: #1E1E1E;
            }
            QTabBar::tab {
                background: #252526;
                color: #D4D4D4;
                padding: 5px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid #3F3F46;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1E1E1E;
                border-bottom-color: #1E1E1E;
            }
            QTabBar::close-button {
                image: url(close.png);
                subcontrol-position: right;
            }
        """)


    def load_api_keys(self):
        encrypted = self.settings.get('api_keys', {})
        for key, value in encrypted.items():
            try:
                decrypted = self.encryption_manager.decrypt(value)
                self.key_combo.addItem(key, decrypted)
            except Exception as e:
                logging.error(f"API kulcs dekódolási hiba: {str(e)}")
        if self.key_combo.count() > 0:
            self.key_combo.setCurrentIndex(0)


    def add_api_key(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Új API kulcs hozzáadása")
        layout = QFormLayout(dialog)


        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Kulcs neve")
        layout.addRow("Név:", name_edit)


        key_edit = QLineEdit()
        key_edit.setPlaceholderText("API kulcs")
        layout.addRow("Kulcs:", key_edit)


        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)


        if dialog.exec_() == QDialog.Accepted:
            name = name_edit.text().strip()
            key = key_edit.text().strip()
            if name and key:
                encrypted = self.encryption_manager.encrypt(key)
                api_keys = self.settings.get('api_keys', {})
                api_keys[name] = encrypted
                self.settings.set('api_keys', api_keys)
                self.key_combo.addItem(name, key)
                self.key_combo.setCurrentText(name)


    def load_settings(self):
        try:
            size_str = self.settings.get('window_size', '1000x700')
            width, height = map(int, size_str.split('x'))
            self.resize(QSize(width, height))
        except (ValueError, TypeError):
            self.resize(1000, 700)


        self.move(QApplication.desktop().screenGeometry().center() - self.frameGeometry().center())
        self.temp_spin.setValue(float(self.settings.get('temperature', 0.4)))
        default_tokens_index = TOKEN_OPTIONS.index(4096) if 4096 in TOKEN_OPTIONS else 3
        tokens_index = TOKEN_OPTIONS.index(int(self.settings.get('max_tokens', 4096))) if int(
            self.settings.get('max_tokens', 4096)) in TOKEN_OPTIONS else default_tokens_index
        self.token_combo.setCurrentIndex(tokens_index)
        self.free_check.setChecked(self.settings.get('free_models', 'true') == 'true')


    def save_settings(self):
        self.settings.set('window_size', f"{self.width()}x{self.height()}")
        self.settings.set('temperature', self.temp_spin.value())
        self.settings.set('max_tokens', self.token_combo.currentData())
        self.settings.set('free_models', self.free_check.isChecked())


    def closeEvent(self, event):
        self.autosave_history()
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()


        tracemalloc.stop()
        super().closeEvent(event)


    def refresh_models(self):
        self.model_combo.clear()
        self.network_manager.free_only = self.free_check.isChecked()
        self.network_manager.models_loaded.connect(self.populate_models)
        self.network_manager.error_occurred.connect(self.show_error)
        self.network_manager.start()


    def populate_models(self, models: List[str]):
        self.model_combo.addItems(models)
        if self.model_combo.count() > 0:
            last_model = self.settings.get('last_model')
            if last_model and last_model in models:
                index = models.index(last_model)
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.setCurrentIndex(0)


    def upload_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Fájl feltöltése", "",
                                                  "Minden fájl (*);;Szövegfájlok (*.txt);;Kód fájlok (*.py *.c *.cpp *.java)",
                                                  options=options)
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > MAX_FILE_SIZE:
                    QMessageBox.warning(self, "Túl nagy fájl",
                                        f"A fájl mérete ({len(content)} karakter) meghaladja a maximális {MAX_FILE_SIZE} karaktert.")
                    return


                self.input_edit.setPlainText(f"A következő kód van feltöltve:\n```plaintext\n{content}\n```\n\nKérés:")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Fájl olvasási hiba: {str(e)}")


    def send_request(self):
        self.start_request(continue_conversation=False)


    def continue_request(self):
        self.start_request(continue_conversation=True)


    def start_request(self, continue_conversation: bool = False):
        self.current_prompt = self.input_edit.toPlainText().strip()
        if not self.current_prompt and not continue_conversation:
            QMessageBox.warning(self, "Üres kérés", "Kérlek, írj be egy kérdést!")
            return


        api_key = self.key_combo.currentData()
        if not api_key:
            QMessageBox.warning(self, "Hiányzó API kulcs", "Kérlek, add meg az API kulcsot!")
            return


        model = self.model_combo.currentText()
        if not model:
            QMessageBox.warning(self, "Hiányzó modell", "Kérlek, válassz egy modellt!")
            return


        self.set_ui_state(False)
        self.is_generating = True  # Jelzés, hogy válasz generálása folyamatban van
        if not continue_conversation:
            self.clear_chat_display()
            self.history.append({"role": "user", "content": self.current_prompt})


            self.append_to_chat(f"**Felhasználó:** {self.current_prompt}\n\n\n***\n", role="user")


        self.status_bar.showMessage("Kérés folyamatban...")
        self.start_worker(api_key, model)
        self.set_generating_background(True)  # Zöld háttér beállítása


    def set_ui_state(self, enabled: bool):
        self.send_btn.setEnabled(enabled)
        self.cont_btn.setEnabled(enabled)
        self.upload_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)


    def start_worker(self, api_key: str, model: str):
        self.worker = AIWorker(
            api_key,
            self.history,
            model,
            self.temp_spin.value(),
            self.token_combo.currentData()
        )
        self.worker.update_received.connect(self.handle_update)
        self.worker.response_completed.connect(self.request_completed)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.truncated.connect(self.show_truncated_message)
        self.worker.start()


    def append_to_chat(self, text: str, role: str = None):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)


        # Szövegformázás beállítása a szerep alapján
        format = QTextCharFormat()
        if role == "user":
            format.setForeground(QColor("lightblue"))  # Felhasználói kérés színe
        elif self.is_generating:
            format.setForeground(QColor("lightgreen"))  # Válasz generálás közben


        cursor.insertText(text, format)
        self.chat_display.ensureCursorVisible()
        # Az append_to_chat végén hívjuk meg a process_code_blocks-ot
        self.process_code_blocks()


    def request_completed(self, status: str):
        if self.buffered_text:
            self.text_receiver.update_text.emit(self.buffered_text)  # Jelzés küldése
            self.buffered_text = ""
        self.history.append({"role": "assistant",
                             "content": self.chat_display.toPlainText().split('Felhasználó:')[-1].strip()})
        self.set_ui_state(True)
        self.status_bar.showMessage(status)
        self.update_history_menu()
        self.settings.set('last_model', self.model_combo.currentText())
        self.autosave_history()
        self.is_generating = False  # Válasz generálás befejeződött
        self.set_generating_background(False)  # Zöld háttér eltávolítása

        # Törli a kérést a kérés ablakból
        self.input_edit.clear()

        # Jelzi a chat ablak függőleges görgetősávjának háttérszínét
        self.set_scroll_indicator_color(QColor("lightgreen"))


    def show_error(self, message: str, status_code: int = None):
        error_message = f"Hiba: {message}"
        if status_code:
            error_message += f" (Státusz kód: {status_code})"
        QMessageBox.critical(self, "Hiba", error_message)
        self.status_bar.showMessage(f"Hiba: {message}")
        self.set_ui_state(True)
        self.is_generating = False  # Válasz generálás befejeződött
        self.set_generating_background(False)  # Zöld háttér eltávolítása


    def show_truncated_message(self):
        QMessageBox.warning(self, "Csonkolt válasz",
                            "A válasz csonkolva lett a maximális tokenek számának elérése miatt.")


    def clear_chat_display(self):
        self.chat_display.clear()
        self.code_blocks = []
        # Kód tabok törlése, de a Chat tab megmarad
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        self.code_tab_count = 0


    def save_chat(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Chat mentése", "", "JSON fájlok (*.json);;Minden fájl (*)",
                                                   options=options)
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, ensure_ascii=False, indent=2)
                self.status_bar.showMessage(f"Chat mentve: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Mentési hiba", f"Nem sikerült menteni a chatet: {str(e)}")


    def load_chat(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Chat betöltése", "", "JSON fájlok (*.json);;Minden fájl (*)",
                                                   options=options)
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                self.clear_chat_display()
                for message in self.history:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    self.append_to_chat(f"**{role.capitalize()}:** {content}\n\n", role=role)  # Szerepet is átadjuk
                self.process_code_blocks()
                self.status_bar.showMessage(f"Chat betöltve: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Betöltési hiba", f"Nem sikerült betölteni a chatet: {str(e)}")


    def update_history_menu(self):
        self.history_menu.clear()
        history_files = []
        try:
            history_files = sorted([f for f in os.listdir(self.settings.history_dir) if f.endswith('.json')])
        except Exception as e:
            logging.error(f"Előzmény fájlok listázása hiba: {str(e)}")


        for filename in history_files[-MAX_HISTORY:]:
            action = self.history_menu.addAction(filename)
            action.triggered.connect(lambda checked=False, f=filename: self.load_history_file(f))


        if not history_files:
            self.history_menu.addAction("Nincs előzmény")


    def load_history_file(self, filename: str):
        filepath = os.path.join(self.settings.history_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            self.clear_chat_display()
            for message in self.history:
                role = message.get('role', 'user')
                content = message.get('content', '')
                self.append_to_chat(f"**{role.capitalize()}:** {content}\n\n", role=role)  # Szerepet is átadjuk
            self.process_code_blocks()
            self.status_bar.showMessage(f"Előzmény betöltve: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Előzmény betöltési hiba", f"Nem sikerült betölteni az előzményt: {str(e)}")


    def clear_history(self):
        reply = QMessageBox.question(self, 'Előzmények törlése',
                                     "Biztosan törlöd az összes előzményt?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)


        if reply == QMessageBox.Yes:
            try:
                for filename in os.listdir(self.settings.history_dir):
                    file_path = os.path.join(self.settings.history_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Törli a fájlt
                self.history = []  # Törli a memóriában tárolt előzményeket
                self.update_history_menu()  # Frissíti az előzmény menüt
                QMessageBox.information(self, "Előzmények törlése", "Az összes előzmény sikeresen törölve.")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Hiba történt az előzmények törlése közben: {str(e)}")


    def process_code_blocks(self):
        text = self.chat_display.toPlainText()
        matches = re.finditer(r"```([a-zA-Z]{3,})\n(.*?)```", text, re.DOTALL)
        for match in matches:
            language = match.group(1).strip() if match.group(1) else "plaintext"
            code = match.group(2).strip()
            self.add_code_tab(language, code)


    def copy_code(self):
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index <= 0:
            return


        widget = self.tab_widget.widget(current_tab_index)
        if isinstance(widget, CodeEditor):
            code = widget.text()
            QApplication.clipboard().setText(code)
            self.status_bar.showMessage("Kód másolva a vágólapra!")


    def close_tab(self, index):
        widget = self.tab_widget.widget(index)
        if widget:
            widget.deleteLater()
        self.tab_widget.removeTab(index)
        self.update_copy_button_state(self.tab_widget.currentIndex())


    def add_code_tab(self, language, code):
        # Ellenőrizd, hogy van-e már ilyen kód a tabokban
        for i in range(1, self.tab_widget.count()):  # Kezdj 1-től, mert a 0. a Chat tab
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.text() == code:
                return  # Ha már van, ne adjuk hozzá újra


        code_editor = CodeEditor()
        code_editor.set_language(language)
        code_editor.setText(code)
        self.code_tab_count += 1
        self.tab_widget.addTab(code_editor, f"Kód {self.code_tab_count} ({language})")


    def search_chat(self, text):
        if not text:
            self.reset_chat_formatting()
            return


        self.reset_chat_formatting()
        text_cursor = self.chat_display.textCursor()
        original_position = text_cursor.position()


        format = QTextCharFormat()
        format.setBackground(QColor("yellow"))  # Kiemelés színe


        self.chat_display.moveCursor(QTextCursor.Start)
        while self.chat_display.find(text):
            self.chat_display.textCursor().mergeCharFormat(format)


        self.chat_display.setTextCursor(text_cursor)
        text_cursor.setPosition(original_position)


    def reset_chat_formatting(self):
        text_cursor = self.chat_display.textCursor()
        text_cursor.select(QTextCursor.Document)
        format = QTextCharFormat()
        format.setBackground(Qt.transparent)
        text_cursor.mergeCharFormat(format)
        text_cursor.clearSelection()
        self.chat_display.setTextCursor(text_cursor)


    def set_generating_background(self, is_generating):
        """
        Beállítja a chat_display háttérszínét a válaszadás állapotától függően.
        """
        if is_generating:
            self.chat_display.setStyleSheet("background-color: #304030;")  # Zöldes háttér
        else:
            self.chat_display.setStyleSheet("background-color: #1E1E1E;")  # Alapértelmezett háttér


    def set_scroll_indicator_color(self, color):
        """
        Beállítja a chat ablak függőleges görgetősávjának háttérszínét.
        """
        # A stíluslap módosítása a függőleges görgetősáv háttérszínének beállításához
        stylesheet = f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
            }}
            QScrollBar:vertical {{
                background-color: #252526;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {color.name()};
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical {{
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }}
            QScrollBar::sub-line:vertical {{
                height: 0 px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }}
        """
        self.chat_display.setStyleSheet(stylesheet)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_()) 

    #--hidden-import=cryptography --hidden-import=cryptography.fernet --hidden-import=psutil --hidden-import=aiohttp --hidden-import=asyncio --hidden-import=PyQt5.sip --hidden-import=PyQt5.QtCore --hidden-import=PyQt5.QtGui --hidden-import=PyQt5.QtWidgets --hidden-import=PyQt5.Qsci