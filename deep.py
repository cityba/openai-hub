import sys
import os
import json
import time
import platform
import re
import asyncio
import aiohttp
import requests
import psutil
from typing import List, Dict
from cryptography.fernet import Fernet
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QTabWidget, QCheckBox, QSplitter,
    QMessageBox, QToolBar, QAction, QStatusBar, QFileDialog, QMenu, QToolButton,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QShortcut, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer, QObject, QSize,QPropertyAnimation
from PyQt5.QtGui import QTextCursor, QPalette, QColor, QFont, QIcon, QTextCharFormat, QKeySequence, QTextDocument,QLinearGradient,QBrush

# Scintilla lexer import
try:
    from PyQt5.Qsci import (QsciScintilla, QsciLexerPython, QsciLexerCPP, QsciLexerJava, QsciLexerJavaScript,
                           QsciLexerHTML, QsciLexerXML, QsciLexerJSON, QsciLexerBash, QsciLexerSQL)
    HAS_SCINTILLA = True
except ImportError:
    HAS_SCINTILLA = False

# Alapbeállítások
APP_NAME = "SzitaAIPro"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_URL = "https://openrouter.ai/api/v1/models"
MAX_HISTORY = 15
MAX_FILE_SIZE = 30000
TOKEN_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_TEMP = 0.4
DEFAULT_TOKENS_INDEX = 3

def optimize_system():
    """Rendszerrősszék optimalizálása"""
    try:
        proc = psutil.Process(os.getpid())
        if platform.system() == "Windows":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(-18)

        num_threads = str(os.cpu_count() or 4)
        os.environ["OMP_NUM_THREADS"] = num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["MKL_NUM_THREADS"] = num_threads
    except Exception:
        pass

optimize_system()

class SettingsManager:
    """Beállítások kezelése"""
    def __init__(self):
        self.config_dir = os.path.join(os.getenv('APPDATA', os.path.expanduser("~")), APP_NAME)
        os.makedirs(self.config_dir, exist_ok=True)
        self.settings = QSettings(os.path.join(self.config_dir, 'config.ini'), QSettings.IniFormat)

    def get(self, key: str, default=None):
        return self.settings.value(key, default)

    def set(self, key, value):
        self.settings.setValue(key, value)

    @property
    def history_dir(self):
        path = os.path.join(self.config_dir, 'history')
        os.makedirs(path, exist_ok=True)
        return path

settings = SettingsManager()

class EncryptionManager:
    """Titkosítás kezelése"""
    def __init__(self):
        key = settings.get('encryption_key')
        if not key:
            key = Fernet.generate_key().decode()
            settings.set('encryption_key', key)
        self.cipher = Fernet(key.encode())

    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        try:
            return self.cipher.decrypt(data.encode()).decode()
        except Exception:
            return ""

encryptor = EncryptionManager()

class NetworkManager(QThread):
    """Hálózati kezelés"""
    models_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.free_only = True
        self.session = None

    async def fetch_models(self):
        try:
            async with aiohttp.ClientSession() as session:
                self.session = session
                async with session.get(MODEL_URL, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    parsed = self.parse_models(data.get('data', []))
                    self.models_loaded.emit(parsed)
        except aiohttp.ClientError as e:
            self.error_occurred.emit(f"Network error: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {e}")
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    def run(self):
        asyncio.run(self.fetch_models())

    def parse_models(self, models: List[Dict]) -> List[str]:
        result = []
        providers = {'deepseek', 'openrouter', 'google', 'bigcode', 'mistral', 'meta',
                     'moonshotai', 'anthropic', 'openai', 'nous', 'perplexity', 'qwen'}

        for m in models:
            model_id = m.get('id', '')
            context = m.get('context_length')

            # prompt/completion lehet közvetlenül vagy limits alatt
            limits = m.get('limits', {})
            prompt = m.get('prompt', limits.get('prompt'))
            completion = m.get('completion', limits.get('completion'))

            is_free = ":free" in model_id

            # csak free modellek, ha kell
            if self.free_only and not is_free:
                continue

            # provider szűrés
            if not any(p in model_id for p in providers):
                continue

            # ha van normális context_length, vagy prompt=0 és completion=0
            if isinstance(context, int):
                tokens = context // 1024
            elif prompt == 0 and completion == 0:
                tokens = 0
            else:
                continue

            label = f"{model_id} | {tokens}K {'🆓' if is_free else '💲'}"
            result.append(label)

        return sorted(result)

class AIWorker(QThread):
    """AI munkamenet kezelése"""
    update_received = pyqtSignal(str)
    response_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, int)
    truncated = pyqtSignal()

    def __init__(self, api_key: str, messages: List[Dict], model: str,
                 temperature: float, max_tokens: int):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model = model.split('|')[0].strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.running = True
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def run(self):
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
            resp = self.session.post(API_URL, json=payload, stream=True, timeout=60)
            if resp.status_code != 200:
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    err = resp.json().get('error', {}).get('message', 'Unknown error')
                    self.error_occurred.emit(str(e), resp.status_code)
                return

            buffer = bytearray()
            for chunk in resp.iter_lines():
                if not self.running:
                    return
                if chunk:
                    data = chunk[5:].strip() if chunk.startswith(b'data:') else chunk.strip()
                    if not data.startswith(b'{'):
                        continue
                    parsed = json.loads(data.decode('utf-8'))
                    choices = parsed.get('choices', [{}])
                    if choices:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            buffer.extend(content.encode('utf-8'))
                            self.update_received.emit(buffer.decode('utf-8', errors='replace'))
                            buffer.clear()
                        finish_reason = choices[0].get('finish_reason')
                        if finish_reason == 'length':
                            self.truncated.emit()

            if buffer:
                self.update_received.emit(buffer.decode('utf-8', errors='replace'))

            self.response_completed.emit("Kész!")
        except requests.RequestException as e:
            self.error_occurred.emit(f"Network error: {e}", 500)
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {e}", 500)

    def stop(self):
        self.running = False
        self.session.close()

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
try:
    from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciLexerCPP, QsciLexerJava, QsciLexerJavaScript, QsciLexerHTML, QsciLexerXML, QsciLexerJSON, QsciLexerSQL, QsciLexerBash
    HAS_SCINTILLA = True
except ImportError:
    HAS_SCINTILLA = False

class CodeEditor(QWidget):
    """Kódszerkesztő widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        if HAS_SCINTILLA:
            self.editor = QsciScintilla()
            self.editor.setAutoIndent(True)
            self.editor.setIndentationGuides(True)
            self.editor.setMarginLineNumbers(1, True)
            self.editor.setMarginWidth(1, "00000")
            self.editor.setBraceMatching(QsciScintilla.SloppyBraceMatch)
            self.editor.setCaretLineVisible(True)
            self.editor.setCaretLineBackgroundColor(QColor(200, 230, 200))
            
            # Beállítjuk a háttérszínt a QsciScintilla esetén
            self.editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, QsciScintilla.STYLE_DEFAULT, QColor(204, 214, 203))  # Pasztel zöld
        else:
            self.editor = QPlainTextEdit()
            self.editor.setReadOnly(True)

        # A QPlainTextEdit háttérszínének beállítása
        self.editor.setStyleSheet("background-color:#ced6cb;")

        self.layout.addWidget(self.editor)

    def set_language(self, language):
        if not HAS_SCINTILLA:
            return
        lexer_map = {
            "python": QsciLexerPython,
            "cpp": QsciLexerCPP,
            "c++": QsciLexerCPP,
            "java": QsciLexerJava,
            "javascript": QsciLexerJavaScript,
            "js": QsciLexerJavaScript,
            "typescript": QsciLexerJavaScript,
            "ts": QsciLexerJavaScript,
            "php": QsciLexerHTML,
            "html": QsciLexerHTML,
            "xml": QsciLexerXML,
            "json": QsciLexerJSON,
            "sql": QsciLexerSQL,
            "bash": QsciLexerBash,
            "sh": QsciLexerBash
        }
        lexer = lexer_map.get(language.lower())
        if lexer:
            self.editor.setLexer(lexer())
        else:
            print(f"Lexer not found for language: {language}")

    def setText(self, text):
        if HAS_SCINTILLA:
            self.editor.setText(text)
        else:
            self.editor.setPlainText(text)

    def text(self):
        return self.editor.text() if HAS_SCINTILLA else self.editor.toPlainText()
class SearchDialog(QDialog):
    """Keresés dialógusablak"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keresés")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()

        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self.on_search_text_changed)
        self.form.addRow("Keresés:", self.search_edit)

        self.layout.addLayout(self.form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.find_next)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.find_next_btn = QPushButton("Következő")
        self.find_prev_btn = QPushButton("Előző")
        self.buttons.addButton(self.find_next_btn, QDialogButtonBox.ActionRole)
        self.buttons.addButton(self.find_prev_btn, QDialogButtonBox.ActionRole)
        self.find_next_btn.clicked.connect(self.find_next)
        self.find_prev_btn.clicked.connect(self.find_prev)

        self.editor = parent.chat_display
        self.cursor = self.editor.textCursor()

    def on_search_text_changed(self, text):
        self.cursor = self.editor.textCursor()
        self.cursor.setPosition(0)

    def find_next(self):
        self._find(QTextDocument.FindFlags())

    def find_prev(self):
        self._find(QTextDocument.FindBackward)

    def _find(self, flags):
        text = self.search_edit.text()
        if not text:
            return
        cursor = self.editor.document().find(text, self.cursor, flags)
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)
            self.cursor = cursor
        else:
            QMessageBox.information(self, "Keresés", "Nincs több találat.")
            self.cursor.setPosition(0 if flags & QTextDocument.FindBackward else QTextCursor.End)
            self._find(flags)

class TextReceiver(QObject):
    """Szövegkezelő osztály"""
    update_text = pyqtSignal(str)

class MainWindow(QWidget):
    """Főablak osztály"""
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
        self.update_interval = 80
        self.text_receiver = TextReceiver()
        self.text_receiver.update_text.connect(self.append_to_chat)
        self.is_generating = False
        self.code_tab_count = 0

        self.setup_ui()
        self.setup_connections()
        self.load_settings()
        self.setWindowIcon(self.get_application_icon())
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.flush_buffer)

    def setup_ui(self):
        """Felület létrehozása"""
        self.setWindowTitle("Szita AI Kódasszisztens")
        self.setMinimumSize(1000, 700)

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
        key_layout.addWidget(QLabel("Kulcs:"))
        key_layout.addWidget(self.key_combo)
        api_layout.addLayout(key_layout)

        model_layout = QHBoxLayout()
        self.free_check = QCheckBox("Csak Ingyenes")
        self.free_check.setChecked(True)
        model_layout.addWidget(self.free_check)
        model_layout.addWidget(QLabel("Modell:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_layout.setStretch(2, 1)
        api_layout.addLayout(model_layout)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Hőmérséklet:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(DEFAULT_TEMP)
        param_layout.addWidget(self.temp_spin)

        param_layout.addWidget(QLabel("Max tokenek:"))
        self.token_combo = QComboBox()
        for t in TOKEN_OPTIONS:
            self.token_combo.addItem(f"{t // 1024}K", t)
        self.token_combo.setCurrentIndex(DEFAULT_TOKENS_INDEX)
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
        self.stop_btn = QPushButton("Leállítás")
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.cont_btn)
        btn_layout.addWidget(self.stop_btn)
        input_layout.addLayout(btn_layout)
        left_layout.addWidget(input_group, 1)

        splitter.addWidget(left_panel)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

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
        """Eszköztár létrehozása"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))

        def add_action(icon, text, callback):
            act = QAction(QIcon.fromTheme(icon), text, self)
            act.triggered.connect(callback)
            toolbar.addAction(act)

        add_action('document-save', "Mentés", self.save_chat)
        add_action('document-open', "Betöltés", self.load_chat)
        add_action('edit-clear', "Törlés", self.clear_chat_display)
        add_action('edit-clear', "Előzmények törlése", self.clear_history)

        toolbar.addSeparator()

        self.history_menu = QMenu("Előzmények", self)
        menu_btn = QToolButton()
        menu_btn.setText("Előzmények")
        menu_btn.setMenu(self.history_menu)
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(menu_btn)

        key_menu = QMenu("Kulcsok", self)
        add_action = QAction("Új kulcs hozzáadása", self)
        add_action.triggered.connect(self.add_api_key)
        key_menu.addAction(add_action)
        toolbar.addAction(key_menu.menuAction())

        self.update_history_menu()
        return toolbar

    def setup_connections(self):
        """Kapcsolatok létrehozása"""
        self.send_btn.clicked.connect(self.send_request)
        self.stop_btn.clicked.connect(self.stop_request)
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
        """Keresés dialógus megjelenítése"""
        self.search_dialog = SearchDialog(self)
        self.search_dialog.show()

    def update_copy_button_state(self, index):
        """Másolás gomb állapotának frissítése"""
        self.copy_btn.setEnabled(index > 0)

    def apply_dark_theme(self):
        """Sötét téma alkalmazása"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#2c3e50"))
        palette.setColor(QPalette.WindowText, QColor("#ecf0f1"))
        palette.setColor(QPalette.Base, QColor("#34495e"))
        palette.setColor(QPalette.AlternateBase, QColor("#2c3e50"))
        palette.setColor(QPalette.ToolTipBase, QColor("#34495e"))
        palette.setColor(QPalette.ToolTipText, QColor("#ecf0f1"))
        palette.setColor(QPalette.Text, QColor("#ecf0f1"))
        palette.setColor(QPalette.Button, QColor("#3498db"))
        palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
        palette.setColor(QPalette.Highlight, QColor("#3498db"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

        app.setStyle("Fusion")
        self.setPalette(palette)
        self.setStyleSheet("""
            QWidget { background-color:#2c3e50; color:#ecf0f1; font-family:"Segoe UI"; font-size:14px; }
            QTextEdit, QPlainTextEdit { background:#34495e; color:#ecf0f1; border:1px solid #2c3e50; border-radius:8px; padding:12px; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background:#34495e; color:#ecf0f1; border:1px solid #2c3e50; border-radius:8px; padding:8px; }
            QPushButton { background:#3498db; color:white; padding:10px 20px; border-radius:8px; border:none; font-weight:bold; }
            QPushButton:hover { background:#2980b9; }
            QPushButton:pressed { background:#2471a3; }
            QPushButton:disabled { background:#7f8c8d; color:#bdc3c7; }
            QStatusBar { color:#95a5a6; font-size:12px; background:#232f34; border-top:1px solid #2c3e50; }
            QTabWidget::pane { border:1px solid #2c3e50; background:#232f34; border-radius:8px; }
            QTabBar::tab { background:#34495e; color:#ecf0f1; padding:8px 16px; border-top-left-radius:8px; border-top-right-radius:8px; }
            QTabBar::tab:selected { background:#2c3e50; }
            QGroupBox { border:1px solid #2c3e50; border-radius:8px; margin-top:1em; padding:10px; }
            QGroupBox::title { left:10px; padding:0 3px; color:#ecf0f1; }
            QToolBar { background:#232f34; padding:5px; }
            QToolButton { background:transparent; border:none; padding:5px; color:#ecf0f1; }
            QToolButton:hover { background:#34495e; border-radius:4px; }
            QMenu { background:#34495e; color:#ecf0f1; border:1px solid #2c3e50; border-radius:4px; }
            QMenu::item { padding:8px 20px; }
            QMenu::item:selected { background:#3498db; }
            QScrollBar:vertical { background:#232f34; width:10px; }
            QScrollBar::handle:vertical { background:#3498db; min-height:20px; border-radius:5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
        """)

    def load_settings(self):
        """Beállítások betöltése"""
        try:
            size = self.settings.get('window_size', '1000x700')
            w, h = map(int, size.split('x'))
            self.resize(w, h)
        except Exception:
            self.resize(1000, 700)

        self.move(QApplication.desktop().screenGeometry().center() - self.frameGeometry().center())
        self.temp_spin.setValue(float(self.settings.get('temperature', DEFAULT_TEMP)))
        tokens = int(self.settings.get('max_tokens', 4096))
        idx = next((i for i, t in enumerate(TOKEN_OPTIONS) if t == tokens), DEFAULT_TOKENS_INDEX)
        self.token_combo.setCurrentIndex(idx)
        self.free_check.setChecked(self.settings.get('free_models', 'true') == 'true')

    def save_settings(self):
        """Beállítások mentése"""
        self.settings.set('window_size', f"{self.width()}x{self.height()}")
        self.settings.set('temperature', self.temp_spin.value())
        self.settings.set('max_tokens', self.token_combo.currentData())
        self.settings.set('free_models', self.free_check.isChecked())

    def closeEvent(self, event):
        """Ablak bezárásakor"""
        self.autosave_history()
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)

    def autosave_history(self):
        """Automatikus előzmények mentése"""
        if not self.history:
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        fn = os.path.join(self.settings.history_dir, f"autosave_{ts}.json")
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_history_menu(self):
        """Előzmények menü frissítése"""
        self.history_menu.clear()
        files = sorted([f for f in os.listdir(self.settings.history_dir) if f.endswith('.json')])
        for f in files[-MAX_HISTORY:]:
            act = self.history_menu.addAction(f)
            act.triggered.connect(lambda _, fn=f: self.load_history_file(fn))
        if not files:
            self.history_menu.addAction("Nincs előzmény")

    def load_history_file(self, filename):
        """Előzmény betöltése"""
        path = os.path.join(self.settings.history_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            self.clear_chat_display()
            for m in self.history:
                role = m.get('role', 'user')
                content = m.get('content', '')
                self.append_to_chat(f"**{role.capitalize()}:** {content}\n\n", role=role)
            self.process_code_blocks()
            self.status_bar.showMessage(f"Előzmény betöltve: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Előzmény betöltési hiba", str(e))

    def clear_history(self):
        """Előzmények törlése"""
        reply = QMessageBox.question(self, "Előzmények törlése",
                                     "Biztosan törlöd az összes előzményt?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                for f in os.listdir(self.settings.history_dir):
                    p = os.path.join(self.settings.history_dir, f)
                    if os.path.isfile(p):
                        os.unlink(p)
                self.history = []
                self.update_history_menu()
                QMessageBox.information(self, "Törlés", "Az összes előzmény törölve.")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", str(e))

    def load_api_keys(self):
        """API kulcsok betöltése"""
        data = self.settings.get('api_keys', {})
        self.key_combo.clear()
        for name, enc in data.items():
            try:
                dec = self.encryption_manager.decrypt(enc)
                if dec:
                    self.key_combo.addItem(name, dec)
            except Exception:
                pass
        if self.key_combo.count():
            self.key_combo.setCurrentIndex(0)

    def add_api_key(self):
        """Új API kulcs hozzáadása"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Új API kulcs")
        l = QFormLayout(dlg)
        name_edit = QLineEdit()
        key_edit = QLineEdit()
        l.addRow("Név:", name_edit)
        l.addRow("Kulcs:", key_edit)
        btn = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn.accepted.connect(dlg.accept)
        btn.rejected.connect(dlg.reject)
        l.addRow(btn)
        if dlg.exec_() == QDialog.Accepted:
            name = name_edit.text().strip()
            key = key_edit.text().strip()
            if name and key:
                data = self.settings.get('api_keys', {})
                data[name] = self.encryption_manager.encrypt(key)
                self.settings.set('api_keys', data)
                self.load_api_keys()
                self.key_combo.setCurrentText(name)

    def refresh_models(self):
        """Modellek frissítése"""
        try:
            self.network_manager.models_loaded.disconnect(self.populate_models)
        except TypeError:
            pass
        self.model_combo.clear()
        self.network_manager.free_only = self.free_check.isChecked()
        self.network_manager.models_loaded.connect(self.populate_models)
        self.network_manager.error_occurred.connect(self.show_error)
        self.network_manager.start()

    def populate_models(self, models):
        """Modellek betöltése"""
        self.model_combo.addItems(models)
        if self.model_combo.count():
            last = self.settings.get('last_model')
            if last and last in models:
                self.model_combo.setCurrentIndex(models.index(last))
            else:
                self.model_combo.setCurrentIndex(0)

    def upload_file(self):
        """Fájl feltöltése"""
        fp, _ = QFileDialog.getOpenFileName(self, "Fájl feltöltés",
                                            "", "Minden fájl (*);;Szövegfájlok (*.txt);;Kód (*.py *.c *.cpp *.java)",
                                            options=QFileDialog.Options())
        if fp:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    txt = f.read()
                if len(txt) > MAX_FILE_SIZE:
                    QMessageBox.warning(self, "Túl nagy fájl",
                                        f"Fájl mérete ({len(txt)} karakter) meghaladja a {MAX_FILE_SIZE} karaktert.")
                    return
                self.input_edit.setPlainText(
                    f"A következő kód van feltöltve:\n```plaintext\n{txt}\n```\n\nKérés:")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Fájl olvasási hiba: {e}")

    def send_request(self):
        """Kérés küldése"""
        self.start_request(False)

    def continue_request(self):
        """Kérés folytatása"""
        self.start_request(True)

    def start_request(self, continue_conv: bool):
        """Kérés kezdeményezése"""
        self.current_prompt = self.input_edit.toPlainText().strip()
        if not self.current_prompt and not continue_conv:
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
        self.is_generating = True
        if not continue_conv:
            self.history.append({"role": "user", "content": self.current_prompt})
            self.append_to_chat(f"**Felhasználó:** {self.current_prompt}\n\n", role="user")
        else:
            last_resp = self.history[-1]["content"] if self.history and self.history[-1]["role"] == "assistant" else ""
            self.history.append({
                "role": "user",
                "content": f"\nKérlek komment nélkül folytasd a kódot!\n"
            })

        self.status_bar.showMessage("Kérés folyamatban…")
        self.start_worker(api_key, model)
        self.set_generating_background(True)

    def set_ui_state(self, enabled: bool):
        """Felület állapotának beállítás"""
        self.send_btn.setEnabled(enabled)
        self.cont_btn.setEnabled(enabled)
        self.upload_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def start_worker(self, api_key, model):
        """Munkamenet indítása"""
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

    def handle_update(self, text: str):
        """Válaszkezelés"""
        self.buffered_text += text
        if not self.update_timer.isActive():
            self.update_timer.start(self.update_interval)

    def flush_buffer(self):
        """Pufferválasz kiürítése"""
        if self.buffered_text:
            self.text_receiver.update_text.emit(self.buffered_text)
            self.buffered_text = ""
        if self.worker and (not self.worker.isRunning() or not self.is_generating):
            self.update_timer.stop()

    def append_to_chat(self, text: str, role: str = None):
        """Szöveg hozzáadása a chathez"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        if role == "user":
            fmt.setForeground(QColor("#3498db"))
        elif self.is_generating:
            fmt.setForeground(QColor("#2ecc71"))
        cursor.insertText(text, fmt)
        self.chat_display.ensureCursorVisible()
        self.process_code_blocks()

    def request_completed(self, status: str):
        """Kérés befejezése"""
        if self.buffered_text:
            self.text_receiver.update_text.emit(self.buffered_text)
            self.buffered_text = ""
        assistant_text = self.chat_display.toPlainText().split("Felhasználó:")[-1].strip()
        self.history.append({"role": "assistant", "content": assistant_text})
        self.set_ui_state(True)
        self.status_bar.showMessage(status)
        self.update_history_menu()
        self.settings.set('last_model', self.model_combo.currentText())
        self.autosave_history()
        self.is_generating = False
        self.set_generating_background(False)
        self.input_edit.clear()
         

    def show_error(self, msg: str, code: int = None):
        """Hiba megjelenítése"""
        e = f"Hiba: {msg}"
        if code:
            e += f" (Státusz: {code})"
        QMessageBox.critical(self, "Hiba", e)
        self.status_bar.showMessage(e)
        self.set_ui_state(True)
        self.is_generating = False
        self.set_generating_background(False)

    def show_truncated_message(self):
        """Válasz folytatása"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Folytatás...")
        dlg.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        v = QVBoxLayout()
        lbl = QLabel("A válasz folytatódik…")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size:16px; padding:20px;")
        v.addWidget(lbl)
        dlg.setLayout(v)
        dlg.setFixedSize(300, 100)
        QTimer.singleShot(3000, dlg.close)
        QTimer.singleShot(3000, self.continue_request)
        dlg.exec_()

    def clear_chat_display(self):
        """Chat törlése"""
        self.chat_display.clear()
        self.code_blocks = []
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        self.code_tab_count = 0

    def save_chat(self):
        """Chat mentése"""
        fn, _ = QFileDialog.getSaveFileName(self, "Chat mentése",
                                            "", "JSON fájl (*.json);;Minden fájl (*)",
                                            options=QFileDialog.Options())
        if fn:
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, ensure_ascii=False, indent=2)
                self.status_bar.showMessage(f"Chat mentve: {fn}")
            except Exception as e:
                QMessageBox.critical(self, "Mentési hiba", str(e))

    def load_chat(self):
        """Chat betöltése"""
        fn, _ = QFileDialog.getOpenFileName(self, "Chat betöltése",
                                            "", "JSON fájl (*.json);;Minden fájl (*)",
                                            options=QFileDialog.Options())
        if fn:
            try:
                with open(fn, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                self.clear_chat_display()
                for m in self.history:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    self.append_to_chat(f"**{role.capitalize()}:** {content}\n\n", role=role)
                self.process_code_blocks()
                self.status_bar.showMessage(f"Chat betöltve: {fn}")
            except Exception as e:
                QMessageBox.critical(self, "Betöltési hiba", str(e))

    def process_code_blocks(self):
        """Kódblokkok feldolgozása"""
        text = self.chat_display.toPlainText()
        for m in re.finditer(r"^(```([a-zA-Z]{3,})\n(.*?)\n```)$",
                             text, re.MULTILINE | re.DOTALL):
            lang = m.group(2).strip() or "plaintext"
            code = m.group(3).strip()
            self.add_code_tab(lang, code)

    def add_code_tab(self, lang, code):
        """Új kódfül létrehozása"""
        for i in range(1, self.tab_widget.count()):
            w = self.tab_widget.widget(i)
            if isinstance(w, CodeEditor) and w.text() == code:
                return
        editor = CodeEditor()
        editor.set_language(lang)
        editor.setText(code)
        self.code_tab_count += 1
        self.tab_widget.addTab(editor, f"Kód {self.code_tab_count} ({lang})")

    def copy_code(self):
        """Kód másolása"""
        idx = self.tab_widget.currentIndex()
        if idx <= 0:
            return
        w = self.tab_widget.widget(idx)
        if isinstance(w, CodeEditor):
            QApplication.clipboard().setText(w.text())
            self.status_bar.showMessage("Kód másolva a vágólapra!")

    def close_tab(self, idx):
        """Fül bezárása"""
        w = self.tab_widget.widget(idx)
        if w:
            w.deleteLater()
        self.tab_widget.removeTab(idx)
        self.update_copy_button_state(self.tab_widget.currentIndex())

    def search_chat(self, txt):
        """Chat keresése"""
        if not txt:
            self.reset_chat_formatting()
            return
        self.reset_chat_formatting()
        cursor = self.chat_display.textCursor()
        orig = cursor.position()
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("yellow"))
        self.chat_display.moveCursor(QTextCursor.Start)
        while self.chat_display.find(txt):
            self.chat_display.textCursor().mergeCharFormat(fmt)
        self.chat_display.setTextCursor(cursor)
        cursor.setPosition(orig)

    def reset_chat_formatting(self):
        """Chat formázás visszaállítása"""
        cursor = self.chat_display.textCursor()
        cursor.select(QTextCursor.Document)
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.transparent)
        cursor.mergeCharFormat(fmt)
        cursor.clearSelection()
        self.chat_display.setTextCursor(cursor)

    def set_generating_background(self, is_gen):
        """Állapotfüggő animált háttér beállítása"""
        # Meglévő timer leállítása
        if hasattr(self, 'bg_timer') and self.bg_timer.isActive():
            self.bg_timer.stop()

        # Kezdő és vég színek meghatározása
        start_color = QColor(52, 73, 94)  # #34495e - alap szín
        end_color = QColor(30,  30, 30)   # #54695e - generálás szín
        
        if is_gen:
            self.bg_colors = [start_color, end_color]
            
        else:
            self.bg_colors = [end_color, start_color]  # Vissza az eredeti színre

        self.current_step = 0
        self.steps = 50  # Kevesebb lépés gyorsabb átmenet
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self._update_background_color)
        self.bg_timer.start(20)  # 20 ms -> 50 FPS

    def _update_background_color(self):
        """Háttérszín frissítése az animációhoz"""
        if self.current_step <= self.steps:
            # Szín interpoláció
            start_color = self.bg_colors[0]
            end_color = self.bg_colors[1]
            
            ratio = self.current_step / self.steps
            
            r = int(start_color.red() + (end_color.red() - start_color.red()) * ratio)
            g = int(start_color.green() + (end_color.green() - start_color.green()) * ratio)
            b = int(start_color.blue() + (end_color.blue() - start_color.blue()) * ratio)
            
            color = QColor(r, g, b)
            self.chat_display.setStyleSheet(f"background-color: {color.name()};")
            
            self.current_step += 1
        else:
            # Animáció vége - timer leállítása
            self.bg_timer.stop()
            # Biztosítjuk, hogy a végső szín beállítva legyen
            final_color = self.bg_colors[1]
            self.chat_display.setStyleSheet(f"background-color: {final_color.name()};")
  

    def get_icon_path(self, name):
        """Ikon elérési útja"""
        for p in [os.path.join(os.path.dirname(__file__), name),
                  os.path.join(self.settings.config_dir, name),
                  getattr(sys, "_MEIPASS", "")]:
            if os.path.exists(p):
                return p
        return None

    def get_icon(self, name):
        """Ikon betöltése"""
        path = self.get_icon_path(name)
        return QIcon(path) if path else QIcon()

    def get_application_icon(self):
        """Alkalmazás ikonjának betöltése"""
        return self.get_icon("icon.ico") or self.get_icon("icon.png") or QIcon()

    def stop_request(self):
        """Kérés leállítása"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.set_ui_state(True)
            self.is_generating = False
            self.set_generating_background(False)
            self.status_bar.showMessage("Kérés leállítva.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

    
    #--hidden-import=cryptography --hidden-import=cryptography.fernet --hidden-import=psutil --hidden-import=aiohttp --hidden-import=asyncio --hidden-import=PyQt5.sip --hidden-import=PyQt5.QtCore --hidden-import=PyQt5.QtGui --hidden-import=PyQt5.QtWidgets --hidden-import=PyQt5.Qsci

 