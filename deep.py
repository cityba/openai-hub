import sys
import json
import time
import os
import psutil
import requests
import logging
import platform
from typing import List, Dict

from cryptography.fernet import Fernet

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox,  QTabWidget, QCheckBox, QSplitter,
    QMessageBox, QToolBar, QAction, QStatusBar, QFileDialog, QMenu,  QToolButton,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QSizePolicy, 
    QShortcut, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings,  QSize
from PyQt5.QtGui import (
    QTextCursor, QPalette, QColor, QFont, QIcon, QTextCharFormat,
     QKeySequence,  QTextOption
)

import gc
import tracemalloc

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

# 1. Memória monitorozás indítása
tracemalloc.start()

# 2. Szemétgyűjtés kikényszerítése
gc.collect()

# 3. Objektuméletciklusok monitorozása
objektek = gc.get_objects()
logging.info(f"Aktív objektumok száma: {len(objektek)}")

# 4. Snapshot a memóriahasználatról
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
logging.info("Top 10 memóriahasználati hely:")
for stat in top_stats[:10]:
    logging.info(stat)

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

    def run(self):
        try:
            response = requests.get(MODEL_URL, timeout=6)
            response.raise_for_status()
            parsed = self.parse_models(response.json().get('data', []))
            self.models_loaded.emit(parsed)
        except requests.exceptions.ConnectionError as e:
            self.error_occurred.emit("Hálózati hiba: Nincs internetkapcsolat")
        except Exception as e:
            self.error_occurred.emit(f"Hálózati hiba: {str(e)}")

    def parse_models(self, models: List[Dict]) -> List[str]:
        result = []
        for model in models:
            model_id = model.get('id', '')
            if not any(provider in model_id for provider in ['deepseek', 'openrouter', 'google', 'mistral', 'meta','moonshotai','anthropic']):
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
            "reasoning": { "exclude": True },
            "transforms": ["middle-out"],
            "usage": { "include": True },
            "stream": True
        }

        try:
            response = requests.post(
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

            buffer = ""
            for chunk in response.iter_lines():
                if not self.running:
                    return

                if chunk:
                    try:
                        decoded = chunk.decode('utf-8', errors='replace')
                        if decoded.strip() == '':
                            continue

                        if decoded.startswith('data:'):
                            data = decoded[5:].strip()
                        else:
                            data = decoded.strip()

                        if not data:
                            continue

                        if not data.startswith('{'):
                            logging.debug(f"Non-JSON response: {data}")
                            continue

                        parsed = json.loads(data)
                        choices = parsed.get('choices', [{}])
                        if choices:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                buffer += content

                                if '\n' in content:
                                    self.update_received.emit(buffer)
                                    buffer = ""

                            finish_reason = choices[0].get('finish_reason')
                            if finish_reason == 'length':
                                self.truncated.emit()

                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        if str(e) != "Expecting value: line 1 column 1 (char 0)":
                            logging.error(f"Adat feldolgozási hiba: {str(e)}")
                    except Exception as e:
                        logging.error(f"Váratlan hiba: {str(e)}")

            if buffer:
                self.update_received.emit(buffer)

            self.response_completed.emit("Kész!")

        except requests.RequestException as e:
            self.error_occurred.emit(f"Hálózati hiba: {str(e)}", 500)
        except Exception as e:
            self.error_occurred.emit(f"Váratlan hiba: {str(e)}", 500)

    def stop(self):
        self.running = False

class CodeEditor(QWidget):
    """Kód szerkesztő komponens, amely QScintilla-t használ, ha elérhető, különben egy egyszerű QPlainTextEdit-et."""
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
            self.editor.setWordWrapMode(QTextOption.NoWrap)
            self.editor.setStyleSheet("""
                QPlainTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas;
                    font-size: 10pt;
                }
            """)
        
        self.layout.addWidget(self.editor)

    def set_language(self, language):
        if not HAS_SCINTILLA:
            return
            
        lang = language.lower()
        if lang == "python":
             
            lexer = QsciLexerPython()
            lexer.setDefaultFont(QFont("Consolas", 10))
            lexer.setDefaultPaper(QColor("#1e1e1e"))
            lexer.setDefaultColor(QColor("#cccccc"))
            lexer.setColor(QColor("#6FB7E7"), QsciLexerPython.Keyword)
            lexer.setColor(QColor("#57A64A"), QsciLexerPython.Comment)
            lexer.setColor(QColor("#CE9178"), QsciLexerPython.DoubleQuotedString)
            lexer.setColor(QColor("#D1896C"), QsciLexerPython.SingleQuotedString)
            lexer.setColor(QColor("#96C97B"), QsciLexerPython.Number) 
        elif lang in ["java", "kotlin"]:
            lexer = QsciLexerJava()
        elif lang == "cpp":
            lexer = QsciLexerCPP()
        elif lang == "javascript":
            lexer = QsciLexerJavaScript()
        else:
            lexer = None
            
        if lexer:
            lexer.setDefaultFont(QFont("Consolas", 10))
            lexer.setDefaultColor(QColor(220, 220, 220))
            lexer.setDefaultPaper(QColor(30, 30, 30))
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

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        self.load_settings()
        self.history = []
        self.worker = None
        self.current_prompt = ""
        self.response_buffer = ""
        self.code_blocks = []  # Eltárolja a kódblokkokat (szöveg, nyelv)
        self.setWindowIcon(self.get_application_icon()) 

    def get_icon_path(self, icon_name):
        """Find icon in application directory or AppData folder"""
        # First check in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, icon_name)
        if os.path.exists(local_path):
            return local_path

        # Then check in the AppData directory
        appdata_path = os.path.join(settings.config_dir, icon_name)
        if os.path.exists(appdata_path):
            return appdata_path

        # Finally, try in the executable directory (for bundled apps)
        if hasattr(sys, '_MEIPASS'):
            meipass_path = os.path.join(sys._MEIPASS, icon_name)
            if os.path.exists(meipass_path):
                return meipass_path

        return None

    def get_icon(self, icon_name):
        """Get QIcon object for the specified icon name"""
        icon_path = self.get_icon_path(icon_name)
        if icon_path:
            return QIcon(icon_path)
        return QIcon()

    def get_application_icon(self):
        """Get application icon (prefer .ico, then .png)"""
        # Try .ico file first
        ico_path = self.get_icon_path("icon.ico")
        if ico_path:
            return QIcon(ico_path)

        # Then try .png file
        png_path = self.get_icon_path("icon.png")
        if png_path:
            return QIcon(png_path)

        # Return empty icon if none found
        return QIcon()        
    def autosave_history(self):
        if not self.history:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(settings.history_dir, f"autosave_{timestamp}.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Automatikus mentés hiba: {str(e)}")

    def setup_ui(self):
        self.setWindowTitle("Szita AI Kódasszisztens")
        self.setMinimumSize(1000, 700)
        self.setWindowIcon(self.load_icon())

        main_layout = QVBoxLayout()
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        # Bal oldali panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # API beállítások
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
        self.free_check = QCheckBox("Csak ingyenes modellek")
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
            self.token_combo.addItem(f"{tokens//1024}K", tokens)
        self.token_combo.setCurrentIndex(3)
        param_layout.addWidget(self.token_combo)

        api_layout.addLayout(param_layout)
        left_layout.addWidget(api_group)

        # Bevitel panel
        input_group = QGroupBox("Kérés")
        input_layout = QVBoxLayout(input_group)
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Írd ide a kérdésed...")
        input_layout.addWidget(self.input_edit)

        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Fájl feltöltés")
        self.send_btn = QPushButton("Küldés")
        self.cont_btn = QPushButton("Folytatás")
        self.cont_btn.setEnabled(False)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.cont_btn)
        input_layout.addLayout(btn_layout)
        left_layout.addWidget(input_group, 1)
        splitter.addWidget(left_panel)

        # Jobb oldali panel
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Tab widget létrehozása
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        # Chat tab
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.tab_widget.addTab(self.chat_display, "Chat")

        # Másolás gomb
        self.copy_btn = QPushButton("Kód másolása")
        self.copy_btn.clicked.connect(self.copy_code)
        self.copy_btn.setEnabled(False)

        right_layout.addWidget(self.tab_widget, 1)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addStretch(1)
        right_layout.addLayout(btn_layout)

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

    def update_copy_button_state(self, index):
        """Másolás gomb állapotának frissítése"""
        self.copy_btn.setEnabled(index > 0)  # Csak kód taboknál engedélyezett

    def load_icon(self):
        icon_path = os.path.join(settings.config_dir, 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QIcon()

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
        encrypted = settings.get('api_keys', {})
        for key, value in encrypted.items():
            try:
                decrypted = encryptor.decrypt(value)
                self.key_combo.addItem(key, decrypted)
            except:
                continue

    def save_api_keys(self):
        encrypted = {}
        for i in range(self.key_combo.count()):
            name = self.key_combo.itemText(i)
            key = self.key_combo.itemData(i)
            encrypted[name] = encryptor.encrypt(key)
        settings.set('api_keys', encrypted)

    def add_api_key(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("API Kulcs hozzáadása")
        layout = QFormLayout(dialog)

        name_input = QLineEdit()
        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.Password)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow("Megnevezés:", name_input)
        layout.addRow("Kulcs:", key_input)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text().strip()
            key = key_input.text().strip()
            if name and key:
                self.key_combo.addItem(name, key)
                self.key_combo.setCurrentText(name)
                self.save_api_keys()

    def load_settings(self):
        last_key = settings.get('last_key', '')
        if last_key:
            index = self.key_combo.findText(last_key)
            if index >= 0:
                self.key_combo.setCurrentIndex(index)

        last_model = settings.get('last_model', '')
        if last_model:
            index = self.model_combo.findText(last_model, Qt.MatchContains)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        free_only = settings.get('free_only', 'True') == 'True'
        self.free_check.setChecked(free_only)

        temperature = float(settings.get('temperature', 0.4))
        max_tokens = int(settings.get('max_tokens', 32768))
        self.temp_spin.setValue(temperature)
        token_index = next((i for i, t in enumerate(TOKEN_OPTIONS) if t == max_tokens), 3)
        self.token_combo.setCurrentIndex(token_index)

    def save_settings(self):
        settings.set('last_key', self.key_combo.currentText())
        settings.set('last_model', self.model_combo.currentText())
        settings.set('free_only', str(self.free_check.isChecked()))
        settings.set('temperature', str(self.temp_spin.value()))
        settings.set('max_tokens', str(self.token_combo.currentData()))

    def refresh_models(self):
        self.network_thread = NetworkManager()
        self.network_thread.free_only = self.free_check.isChecked()
        self.network_thread.models_loaded.connect(self.update_models)
        self.network_thread.error_occurred.connect(self.show_error)
        self.network_thread.start()

    def update_models(self, models: List[str]):
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            last_model = settings.get('last_model')
            if last_model:
                index = self.model_combo.findText(last_model, Qt.MatchContains)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                else:
                    self.model_combo.setCurrentIndex(0)
            else:
                self.model_combo.setCurrentIndex(0)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Fájl feltöltése", "", 
            "Összes fájl (*);;Kódfájlok (*.py *.java *.kt *.js *.php);;Excel fájlok (*.xlsx *.xls)"
        )

        if file_path:
            try:
                size = os.path.getsize(file_path)
                if size > MAX_FILE_SIZE:
                    raise IOError("A fájl túl nagy (max 30KB)")

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(MAX_FILE_SIZE)
                    self.input_edit.append(f"\n[Fájl] {os.path.basename(file_path)}:\n```{content[:2000]}...```")
                    self.history.append({
                        'role': 'system',
                        'content': f"Feltöltött fájl: {file_path}\n{content}"
                    })
            except Exception as e:
                self.status_bar.showMessage(f"Fájl hiba: {str(e)}")

    def send_request(self):
        if self.worker and self.worker.isRunning():
            return

        key = self.key_combo.currentData()
        if not key:
            self.status_bar.showMessage("Érvénytelen API kulcs!")
            return

        prompt = self.input_edit.toPlainText().strip()
        if not prompt:
            self.status_bar.showMessage("Írj be egy kérdést!")
            return

        self.current_prompt = prompt
        model = self.model_combo.currentText()
        settings.set('last_model', model)

        messages = [{
            'role': 'system',
            'content': "Professzionális kódoló asszisztens vagy, magyarul beszélsz mindig. Python, kotlin, java, PHP, JavaScript és Excel függvényekre specializálódva. Tiszta, hatékony kódot adj meg a legjobb gyakorlatokkal. Tüntesd fel a szükséges függőségeket és világos magyarázatokat, amikor kérik. Excel esetén képleteket és VBA megoldásokat is adj meg, amikor szükséges."
        }] + self.history + [{'role': 'user', 'content': prompt}]

        self.history.append({'role': 'user', 'content': prompt})
        self.disable_input()
        self.cont_btn.setEnabled(False)
        self.status_bar.showMessage("Kérés küldése...")
        self.append_to_chat(f"\nFelhasználó: {prompt}\n")
        #self.chat_edit.appendHtml( f'<b><span style="color: gray;">Felhasználó: {prompt}</span></b><br>')
        #self.chat_edit.appendHtml( f'<b><span style="color: gray;">Felhasználó: {prompt}</span></b><br>')
        self.response_buffer = ""

        self.worker = AIWorker(
            key,
            messages[-6:],
            model,
            self.temp_spin.value(),
            self.token_combo.currentData()
        )
        self.worker.update_received.connect(self.handle_update)
        self.worker.response_completed.connect(self.handle_completion)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.truncated.connect(self.handle_truncation)
        self.worker.start()

    def continue_request(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Folyamatban", "Már fut egy kérés, kérlek várj.")
            return

        if not self.history or self.history[-1]['role'] != 'assistant':
            QMessageBox.warning(self, "Hiba", "Nincs mit folytatni — az utolsó válasz nem asszisztensi.")
            return

        # Folytatási kérés hozzáadása
        prompt = "Folytasd a választ!"
        self.history.append({'role': 'user', 'content': prompt})
        self.append_to_chat(f"\nFelhasználó: {prompt}\n")
        self.input_edit.setPlainText(prompt)
        self.response_buffer = ""
        self.send_request()

    def handle_update(self, text: str):
        self.response_buffer += text
        
        # Kódblokkok kezelése
        if '```' in self.response_buffer:
            parts = self.response_buffer.split('```', 2)  # Csak az első két elválasztóig
            
            if len(parts) >= 3:
                # Az első rész (szöveg) hozzáadása a chathoz
                if parts[0]:
                    self.append_to_chat(parts[0])
                
                # Kódblokk feldolgozása
                lang_code = parts[1].split('\n', 1)
                lang = lang_code[0].strip() if lang_code else ''
                code = lang_code[1] if len(lang_code) > 1 else parts[1]
                
                # Kódblokk megjelenítése
                self.display_code_block(lang, code)
                
                # Maradék szöveg kezelése
                self.response_buffer = parts[2]
                if self.response_buffer:
                    self.append_to_chat(self.response_buffer)
                    self.response_buffer = ""
            else:
                # Még nincs teljes kódblokk
                self.append_to_chat(text)
        else:
            self.append_to_chat(text)

    def append_to_chat(self, text: str):
        """Szöveg hozzáadása a chat ablakhoz"""
        if text.strip() == '':
            return
            
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Formázás normál szöveghez
        format = QTextCharFormat()
        format.setFont(QFont("Segoe UI", 10))
        format.setForeground(QColor(220, 220, 220))
        cursor.setCharFormat(format)
        
        # Szöveg hozzáadása
        cursor.insertText(text)
        
        # Görgetés az aljára
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def display_code_block(self, language: str, code: str):
        """Kódblokk megjelenítése új tabban"""
        # Egyszerűsített nyelvfelismerés
        lang_map = {
            'py': 'python',
            'python': 'python',
            'js': 'javascript',
            'javascript': 'javascript',
            'java': 'java',
            'kt': 'kotlin',
            'kotlin': 'kotlin',
            'cpp': 'cpp',
            'c++': 'cpp',
            'php': 'php',
            'vba': 'vb',
            'excel': 'vb',
            'vb': 'vb'
        }
        lang_key = language.lower() if language else 'text'
        lang = lang_map.get(lang_key, 'text')

        # Új kód szerkesztő létrehozása
        editor = CodeEditor()
        editor.setText(code)
        editor.set_language(lang)

        # Tab neve
        tab_name = f"Kód: {lang}"

        # Új tab hozzáadása
        tab_index = self.tab_widget.addTab(editor, tab_name)
        self.tab_widget.setCurrentIndex(tab_index)
        self.code_blocks.append((lang, code, tab_index))
        
         
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

    def handle_completion(self, message: str):
        if not self.worker:
            return

        self.history.append({'role': 'assistant', 'content': self.response_buffer})
        self.input_edit.clear()
        self.current_prompt = ""
        self.response_buffer = ""
        self.enable_input()
        self.status_bar.showMessage("Válasz kész! " + message)
        self.save_settings()

    def handle_error(self, error: str, code: int):
        self.enable_input()
        self.cont_btn.setEnabled(False)
        self.status_bar.showMessage(f"Hiba ({code}): {error}")
        if "nameresolutionerror" in error.lower():
            self.status_bar.showMessage("Hálózati hiba: Nem sikerült feloldani a szerver nevét")

    def handle_truncation(self):
        self.cont_btn.setEnabled(True)
        self.status_bar.showMessage("Figyelem! A válasz csonkolva lett")

    def disable_input(self):
        self.input_edit.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.key_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.temp_spin.setEnabled(False)
        self.token_combo.setEnabled(False)
        self.right_panel.setStyleSheet("background-color: #156e1a;")
        self.status_bar.setStyleSheet("background-color: #156e1a;")

    def enable_input(self):
        self.input_edit.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.autosave_history()
        self.upload_btn.setEnabled(True)
        self.key_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.temp_spin.setEnabled(True)
        self.token_combo.setEnabled(True)
        self.status_bar.setStyleSheet("background-color: #1E1E1E;")
        self.right_panel.setStyleSheet("background-color: #1E1E1E;")

    def update_history_menu(self):
        self.history_menu.clear()
        history_dir = settings.history_dir

        clear_action = QAction("Előzmények törlése", self)
        clear_action.triggered.connect(self.clear_history)
        self.history_menu.addAction(clear_action)
        self.history_menu.addSeparator()

        try:
            files = sorted(os.listdir(history_dir),
                          key=lambda f: os.path.getmtime(os.path.join(history_dir, f)),
                          reverse=True)

            for file in files[:15]:
                if file.endswith('.json'):
                    action = QAction(file, self)
                    action.triggered.connect(lambda checked, f=file: self.load_chat(f))
                    self.history_menu.addAction(action)
        except FileNotFoundError:
            pass

    def save_chat(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Beszélgetés mentése", settings.history_dir, "JSON fájlok (*.json)"
        )

        if filename:
            if not filename.endswith('.json'):
                filename += '.json'

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            self.status_bar.showMessage(f"Mentve: {filename}")
            self.update_history_menu()

    def load_chat(self, filename: str = None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Beszélgetés betöltése", settings.history_dir, "JSON fájlok (*.json)"
            )

        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)

                self.chat_display.clear()
                self.tab_widget.clear()
                self.tab_widget.addTab(self.chat_display, "Chat")

                for msg in self.history:
                    if msg.get('role') == 'user':
                        self.append_to_chat(f"\nFelhasználó: {msg.get('content', '')}\n")
                    elif msg.get('role') == 'assistant':
                        self.append_to_chat(f"\nAssistant: {msg.get('content', '')}\n")
                self.status_bar.showMessage(f"Betöltve: {filename}")
            except Exception as e:
                self.show_error(f"Hiba történt a fájl betöltésekor: {str(e)}")

    def clear_history(self):
        reply = QMessageBox.question(
            self,
            "Megerősítés",
            "Biztosan törölni szeretnéd az összes előzményt?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for file in os.listdir(settings.history_dir):
                os.remove(os.path.join(settings.history_dir, file))
            self.history = []
            self.chat_display.clear()
            self.update_history_menu()
            self.status_bar.showMessage("Előzmények törölve!")

    def clear_chat_display(self):
        reply = QMessageBox.question(
            self,
            "Megerősítés",
            "Biztosan törölni szeretnéd a beszélgetést? A művelet nem visszavonható.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            # Csak a kódtabokat zárjuk be
            while self.tab_widget.count() > 1:
                self.tab_widget.removeTab(1)
            self.history = []
            self.status_bar.showMessage("Beszélgetés törölve!")
        
    def copy_code(self):
        """Aktuális kód másolása vágólapra"""
        current_index = self.tab_widget.currentIndex()
        if current_index > 0:  # Az első tab a chat, a többiek kódok
            current_widget = self.tab_widget.currentWidget()
            if isinstance(current_widget, CodeEditor):
                code = current_widget.text()
                QApplication.clipboard().setText(code)
                self.status_bar.showMessage("Kód másolva!", 3000)

    def close_tab(self, index):
        """Tab bezárása, kivéve a chat tabot (index=0)"""
        if index > 0:
            # Kódblokk eltávolítása a listából
            self.code_blocks = [block for block in self.code_blocks if block[2] != index]
            self.tab_widget.removeTab(index)
            
            # Tab indexek frissítése
            for i, block in enumerate(self.code_blocks):
                _, _, tab_idx = block
                if tab_idx > index:
                    self.code_blocks[i] = (block[0], block[1], tab_idx - 1)

    def show_error(self, message: str):
        QMessageBox.critical(self, "Hiba", message)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)

        self.save_settings()
        self.save_api_keys()
        event.accept()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(settings.config_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
#  --hidden-import psutil   --hidden-import _psutil_linux  --hidden-import cryptography.hazmat.bindings.openssl.binding  --hidden-import PyQt5.sip    