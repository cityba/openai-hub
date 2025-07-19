import sys
import json
import time
import re
import os
import psutil
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QFrame, 
    QScrollArea, QTabWidget, QListWidget, QListWidgetItem, 
    QSplitter, QMessageBox, QToolBar, QAction, QStatusBar,
    QFileDialog, QMenu, QMenuBar, QPlainTextEdit, QLineEdit,
    QInputDialog, QDialog, QDialogButtonBox, QFormLayout, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QSize,QTimer
from PyQt5.QtGui import (
    QTextCursor, QPalette, QColor, QFont, QIcon, 
    QTextCharFormat, QBrush, QKeySequence
)


# Optimize resource usage
psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 1)


API_URL = "https://openrouter.ai/api/v1/chat/completions"
CONFIG_DIR = os.path.join(os.getenv('APPDATA'), 'Szita_AI')
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
HISTORY_DIR = os.path.join(CONFIG_DIR, 'history')
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


# System message for professional coding assistance
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Professzionális kódoló asszisztens vagy, magyarul beszélsz mindig. Python, kotlin, java, PHP, JavaScript és Excel függvényekre specializálódva."
        "Tiszta, hatékony kódot adj meg a legjobb gyakorlatokkal. Tüntesd fel a szükséges függőségeket és világos magyarázatokat, amikor kérik."
        "Excel esetén képleteket és VBA megoldásokat is adj meg, amikor szükséges."
    )
}


class Worker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str, int)
    truncated = pyqtSignal()
    stream_interrupted = pyqtSignal()


    def __init__(self, api_key, messages, model, max_tokens, temperature):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.is_running = True
        self.retry_count = 0
        self.full_response = ""
        self.was_truncated = False
        self.model_used = model.split("/")[-1].split(":")[0]
        self.response_started = False
        self.last_received_index = 0


    def run(self):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }
            
            while self.retry_count < 3 and self.is_running:
                try:
                    with requests.post(API_URL, headers=headers, json=data, stream=True, timeout=120) as response:
                        if response.status_code == 429:
                            self.error_occurred.emit("Túl sok kérés. Várj 60 másodpercet...", 429)
                            time.sleep(60)
                            self.retry_count += 1
                            continue
                            
                        if response.status_code != 200:
                            self.error_occurred.emit(f"API Hiba: {response.status_code}", response.status_code)
                            return
                            
                        buffer = ""
                        for chunk in response.iter_lines():
                            if not self.is_running:
                                self.stream_interrupted.emit()
                                return
                                
                            if chunk:
                                decoded_chunk = chunk.decode('utf-8')
                                if decoded_chunk.startswith('data:'):
                                    try:
                                        json_data = json.loads(decoded_chunk[5:].strip())
                                        if "choices" in json_data:
                                            if json_data["choices"][0].get("finish_reason") == "length":
                                                self.was_truncated = True
                                            if json_data["choices"][0].get("delta") and "content" in json_data["choices"][0]["delta"]:
                                                content = json_data["choices"][0]["delta"]["content"]
                                                self.full_response += content
                                                buffer += content
                                                self.last_received_index = len(self.full_response)
                                                # Emit in chunks to reduce UI updates
                                                if len(buffer) > 50 or '\n' in buffer:
                                                    self.chunk_received.emit(buffer)
                                                    buffer = ""
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Emit remaining buffer
                        if buffer:
                            self.chunk_received.emit(buffer)
                            
                        if self.was_truncated:
                            self.truncated.emit()
                        self.finished.emit()
                        return
                        
                except requests.exceptions.Timeout:
                    self.error_occurred.emit("Időtúllépés. Újrapróbálkozás...", 408)
                    self.retry_count += 1
                    time.sleep(5)
                    if self.retry_count >= 3:
                        self.error_occurred.emit("Túl sok próbálkozás. Kérlek próbáld újra később.", 500)
                except requests.exceptions.RequestException as e:
                    self.error_occurred.emit(f"Hálózati hiba: {str(e)}", 500)
                    self.stream_interrupted.emit()
                    return


        except Exception as e:
            self.error_occurred.emit(f"Hiba: {str(e)}", 500)
            self.stream_interrupted.emit()


    def stop(self):
        self.is_running = False


class CodeBlockWidget(QGroupBox):
    def __init__(self, code, language, index, parent=None):
        super().__init__(parent)
        self.setTitle(f"Kódrészlet #{index} ({language})")
        self.setCheckable(True)
        self.setChecked(True)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold; 
                color: #4EC9B0;
                background-color: #1E1E1E;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                margin-top: 10px;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 20, 5, 5)  # Extra top margin for title
        
        # Code editor
        self.code_edit = QTextEdit()
        self.code_edit.setPlainText(code)
        self.code_edit.setReadOnly(True)
        self.code_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.code_edit)
        
        # Copy button
        copy_button = QPushButton("Másolás")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                border: none;
                max-width: 100px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
        """)
        copy_button.clicked.connect(lambda: self.copy_code(code))
        layout.addWidget(copy_button, alignment=Qt.AlignRight)
        
        self.setLayout(layout)
        
        # Animation setup
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(300)
        self.toggled.connect(self.toggle_collapse)
        
        # Store initial height
        self.full_height = self.sizeHint().height()
        self.setMaximumHeight(self.full_height)


    def toggle_collapse(self, checked):
        if checked:
            # Expand
            self.animation.setStartValue(self.minimumHeight())
            self.animation.setEndValue(self.full_height)
        else:
            # Collapse
            self.animation.setStartValue(self.height())
            self.animation.setEndValue(self.minimumHeight())
        self.animation.start()


    def copy_code(self, code):
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        QMessageBox.information(self, "Siker", "Kód másolva a vágólapra!")


class ApiKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Kulcs Hozzáadása")
        layout = QFormLayout(self)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Kulcs megjelenített neve")
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("API kulcs")
        self.key_input.setEchoMode(QLineEdit.Password)
        
        layout.addRow("Megnevezés:", self.name_input)
        layout.addRow("API Kulcs:", self.key_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


    def get_key_data(self):
        return {
            "name": self.name_input.text().strip(),
            "key": self.key_input.text().strip()
        }


class HistoryManager:
    @staticmethod
    def save_history(history, filename=None):
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"history_{timestamp}.json"
        
        file_path = os.path.join(HISTORY_DIR, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            return True, file_path
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def load_history(filename):
        file_path = os.path.join(HISTORY_DIR, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return True, json.load(f)
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def list_history_files():
        files = []
        for file in os.listdir(HISTORY_DIR):
            if file.endswith(".json"):
                files.append(file)
        return sorted(files, reverse=True)
    
    @staticmethod
    def delete_history_file(filename):
        file_path = os.path.join(HISTORY_DIR, filename)
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            return False, str(e)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Szita AI Chat Pro")
        self.setGeometry(100, 100, 1200, 900)
        
        # Set application icon
        self.setWindowIcon(self.get_application_icon())
        
        self.api_keys = {}
        self.current_key = ""
        self.conversation_history = []
        self.current_history_file = None
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        # Menüsor
        menubar = QMenuBar()
        file_menu = menubar.addMenu("Fájl")
        
        save_action = QAction("Mentés", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_conversation)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Mentés másként", self)
        save_as_action.triggered.connect(self.save_conversation_as)
        file_menu.addAction(save_as_action)
        
        load_action = QAction("Betöltés", self)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
        load_action.triggered.connect(self.load_conversation)
        file_menu.addAction(load_action)
        
        upload_action = QAction("Fájl feltöltés", self)
        upload_action.triggered.connect(self.upload_file)
        file_menu.addAction(upload_action)
        
        history_menu = menubar.addMenu("Előzmények")
        self.history_menu = history_menu
        self.update_history_menu()
        
        key_menu = menubar.addMenu("Kulcsok")
        add_key_action = QAction("Kulcs hozzáadása", self)
        add_key_action.triggered.connect(self.add_api_key)
        key_menu.addAction(add_key_action)
        
        exit_action = QAction("Kilépés", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        main_layout.addWidget(menubar)
        
        # Eszköztár
        toolbar = QToolBar()
        toolbar.setMovable(False)
        main_layout.addWidget(toolbar)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # Bal oldali panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # Csökkentett felső margó (eredeti: 10, új: 5)
        left_layout.setContentsMargins(10, 0, 10, 10)  # Felső margó csökkentve

        # Beállítások panel
        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_layout = QVBoxLayout(settings_frame)
        # Jelentősen csökkentett margók a beállítások panelen belül
        settings_layout.setContentsMargins(5, 5, 5, 5)  # Minden irányban kicsi margó
        settings_layout.setSpacing(8)  # Optikailag kellemes térköz

        # API Key választó
        api_key_layout = QHBoxLayout()
        api_key_layout.setContentsMargins(0, 0, 0, 0)  # Nincs margó a HBox körül

        api_key_label = QLabel("API Key:")
        self.api_key_combo = QComboBox()
        self.api_key_combo.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 5px;
            }
        """)

        
        self.api_key_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self.api_key_combo.customContextMenuRequested.connect(self.show_key_context_menu)

        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.api_key_combo, 1)  # 1 = nyújtódjon ki

        settings_layout.addLayout(api_key_layout)
        
         
        # Modell választó
        model_layout = QHBoxLayout()
        model_label = QLabel("Modell:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tngtech/deepseek-r1t2-chimera:free",
            "deepseek/deepseek-r1-0528:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "google/gemini-2.0-flash-exp:free", 
            "deepseek/deepseek-r1-distill-llama-70b:free" 
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)




        
        
        # Paraméterek
        params_layout = QHBoxLayout()
        
        # Temperature
        temp_layout = QVBoxLayout()
        temp_label = QLabel("Kreativitás (temperature):")
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.4)
        self.temp_spin.setDecimals(1)
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_spin)
        params_layout.addLayout(temp_layout)
        
        # Max tokens
        tokens_layout = QVBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        self.tokens_spin = QComboBox()
        self.tokens_spin.addItems(["1024", "2048", "4096", "8192", "16384", "32000", "64000"])
        self.tokens_spin.setCurrentIndex(5)
        tokens_layout.addWidget(tokens_label)
        tokens_layout.addWidget(self.tokens_spin)
        params_layout.addLayout(tokens_layout)
        
        settings_layout.addLayout(params_layout)
        left_layout.addWidget(settings_frame)
        
        # Bevitteli terület
        input_frame = QFrame()
        input_layout = QVBoxLayout(input_frame)
        input_label = QLabel("Prompt:")
        input_layout.addWidget(input_label)
        
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Írd ide a kérdésed vagy promptot...")
        self.input_field.setMinimumHeight(200)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        # Gombok
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Fájl feltöltés")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #6c3483;
                color: white;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
        """)
        self.upload_button.clicked.connect(self.upload_file)
        
        self.send_button = QPushButton("Küldés")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #A0A0A0;
            }
        """)
        self.send_button.clicked.connect(self.send_request)
        
        self.continue_button = QPushButton("Folytatás")
        self.continue_button.setStyleSheet("""
            QPushButton {
                background-color: #388E3C;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #4CAF50;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #A0A0A0;
            }
        """)
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continue_request)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.continue_button)
        input_layout.addLayout(button_layout)
        
        left_layout.addWidget(input_frame, 1)
        splitter.addWidget(left_panel)
        
        # Jobb oldali panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.tab_widget = QTabWidget()
        
        # Válasz fül
        response_tab = QWidget()
        response_layout = QVBoxLayout(response_tab)
        
        self.response_area = QPlainTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
        """)
        self.response_area.setMaximumBlockCount(35000)
        self.response_area.setWordWrapMode(True)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.response_area)
        response_layout.addWidget(scroll_area)
        
         
        
        self.tab_widget.addTab(response_tab, "Válasz")
        
        # Kódrészletek fül
        self.code_tab = QWidget()
        code_layout = QVBoxLayout(self.code_tab)
        
        self.code_scroll = QScrollArea()
        self.code_scroll.setWidgetResizable(True)
        self.code_widget = QWidget()
        self.code_layout = QVBoxLayout(self.code_widget)
        self.code_layout.setAlignment(Qt.AlignTop)
        self.code_scroll.setWidget(self.code_widget)
        
        code_layout.addWidget(self.code_scroll)
        self.tab_widget.addTab(self.code_tab, "Kódrészletek")
        
        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)
        
        # Állapotsor
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                color: #A0A0A0;
                font-size: 12px;
                background-color: #252526;
                border-top: 1px solid #3F3F46;
            }
        """)
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Free modell korlátok: 5 kérés/perc, 100 kérés/nap")
        
        # Kontextusmenü
        self.response_area.setContextMenuPolicy(Qt.CustomContextMenu)
        self.response_area.customContextMenuRequested.connect(self.show_context_menu)
        
        # Worker referencia
        self.worker = None
        self.current_code_blocks = []
        self.last_response = ""
        self.is_continuation = False
        self.first_message = True
        
        # Load settings
        self.load_settings()
        self.set_dark_theme()
        splitter.setSizes([300, 900])


    def get_icon_path(self, icon_name):
        """Find icon in application directory or AppData folder"""
        # First check in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, icon_name)
        if os.path.exists(local_path):
            return local_path
        
        # Then check in the AppData directory
        appdata_path = os.path.join(CONFIG_DIR, icon_name)
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

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        self.setFont(QFont("Segoe UI", 10))


    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.api_keys = data.get('api_keys', {})
                    self.current_key = data.get('current_key', '')
                    
                    self.api_key_combo.clear()
                    for name in self.api_keys:
                        self.api_key_combo.addItem(name)
                    
                    if self.current_key and self.current_key in self.api_keys:
                        index = self.api_key_combo.findText(self.current_key)
                        if index >= 0:
                            self.api_key_combo.setCurrentIndex(index)
            except Exception as e:
                QMessageBox.warning(self, "Hiba", f"Beállítások betöltése sikertelen: {str(e)}")


    def save_settings(self):
        try:
            data = {
                'api_keys': self.api_keys,
                'current_key': self.current_key
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            QMessageBox.warning(self, "Hiba", f"Beállítások mentése sikertelen: {str(e)}")


    def add_api_key(self):
        dialog = ApiKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            key_data = dialog.get_key_data()
            if key_data['name'] and key_data['key']:
                self.api_keys[key_data['name']] = key_data['key']
                self.api_key_combo.addItem(key_data['name'])
                self.current_key = key_data['name']
                self.save_settings()
                self.api_key_combo.setCurrentText(key_data['name'])


    def remove_api_key(self, key_name):
        if key_name in self.api_keys:
            del self.api_keys[key_name]
            index = self.api_key_combo.findText(key_name)
            if index >= 0:
                self.api_key_combo.removeItem(index)
            self.save_settings()


    def show_key_context_menu(self, pos):
        menu = QMenu()
        remove_action = menu.addAction("Kulcs eltávolítása")
        action = menu.exec_(self.api_key_combo.mapToGlobal(pos))
        
        if action == remove_action:
            current_key = self.api_key_combo.currentText()
            if current_key:
                self.remove_api_key(current_key)


    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Fájl feltöltése", "", "Minden fájl (*);;Szövegfájlok (*.txt);;Excel fájlok (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Korlátozzuk az első 2000 karaktert
                preview = content[:2000] + ("..." if len(content) > 2000 else "")
                self.input_field.append(f"\n\n[Feltöltött fájl tartalma - {os.path.basename(file_path)}]:\n{preview}")
                self.status_bar.showMessage(f"Fájl betöltve: {file_path}")
                
                # Tároljuk a fájl elérési útját a kontextusban
                self.conversation_history.append({
                    "role": "system",
                    "content": f"Felhasználó feltöltött egy fájlt: {file_path}"
                })
                
            except UnicodeDecodeError:
                QMessageBox.warning(self, "Hiba", "A fájl nem szöveges formátumú!")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Fájl betöltési hiba: {str(e)}")


    def continue_request(self):
        if not self.conversation_history:
            return
            
        self.is_continuation = True
        self.conversation_history.append({"role": "user", "content": "Folytasd a választ ott, ahol abbahagytad!"})
        self.send_request()


    def send_request(self):
        input_text = self.input_field.toPlainText()
        api_key_name = self.api_key_combo.currentText()
        
        if not api_key_name or api_key_name not in self.api_keys:
            self.response_area.appendPlainText("Kérlek válassz egy érvényes API kulcsot!")
            return
            
        api_key = self.api_keys[api_key_name]
        self.current_key = api_key_name
        self.save_settings()
        
        if not input_text.strip() and not self.is_continuation:
            self.response_area.appendPlainText("Kérlek adj meg egy kérdést!")
            return
            
        self.status_bar.showMessage("Kapcsolódás...")
        
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        
        # Add system message for first request
        if self.first_message:
            self.conversation_history.append(SYSTEM_MESSAGE)
            self.first_message = False
        
        # Kérdés megjelenítése
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Create format for user message
        user_format = QTextCharFormat()
        user_format.setForeground(QColor("#5dade2"))  # Blue color
        user_format.setFontWeight(QFont.Bold)
        
        # Insert user message
        if self.response_area.toPlainText():
            cursor.insertText("\n")
        cursor.insertText("Felhasználó: ", user_format)
        
        # Reset format for actual question
        default_format = QTextCharFormat()
        default_format.setForeground(QColor("#D4D4D4"))
        cursor.setCharFormat(default_format)
        cursor.insertText(f"{input_text}")
        
        # Feldolgozási üzenet hozzáadása
        model_name = self.model_combo.currentText().split("/")[-1].split(":")[0]
        processing_format = QTextCharFormat()
        processing_format.setForeground(QColor("#2ecc71"))  # Light green
        processing_format.setFontWeight(QFont.Bold)

        # Insert AI processing message
        cursor.insertText("\n\n")
        cursor.insertText(f"Szita AI ({model_name}): Kérdés feldolgozása...", processing_format)
                
        # Insert AI processing message
         
        dot = 0
        def dupdate():
            nonlocal dot
            dot = (dot % 3) + 1            
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()            
            cursor.insertText(f"Szita AI ({model_name}): Kérdés feldolgozása" + "."*dot, processing_format)
            

        timer = QTimer()
        timer.timeout.connect(dupdate)
        timer.start(1000)
                
        self.response_area.setTextCursor(cursor)
        if not self.is_continuation:
            self.conversation_history.append({"role": "user", "content": input_text})
        
        self.worker = Worker(
            api_key=api_key,
            messages=self.conversation_history,
            model=self.model_combo.currentText(),
            max_tokens=int(self.tokens_spin.currentText()),
            temperature=self.temp_spin.value()
        )
        
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.finished.connect(self.handle_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.truncated.connect(self.handle_truncated)
        self.worker.stream_interrupted.connect(self.handle_stream_interrupted)
        self.worker.start()

    def handle_chunk(self, chunk):
        if not self.worker.response_started:
            # Töröljük a feldolgozási üzenetet
            cursor = self.response_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            
            # Töröljük a prompt szövegét amikor megérkezik az első válasz
            self.input_field.clear()
            
            model_name = self.worker.model_used
            cursor = self.response_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            # Create format for AI name
            ai_format = QTextCharFormat()
            ai_format.setForeground(QColor("#2ecc71"))  # Light green
            ai_format.setFontWeight(QFont.Bold)
            
            # Insert AI name
            if self.response_area.toPlainText():
                cursor.insertText("\n\n")
            cursor.insertText(f"Szita AI ({model_name}):\n", ai_format)
            
            # Reset to default format for response text
            default_format = QTextCharFormat()
            default_format.setForeground(QColor("#D4D4D4"))
            cursor.setCharFormat(default_format)
            
            self.response_area.setTextCursor(cursor)
            self.worker.response_started = True
        
        # Insert the actual response chunks
        self.response_area.insertPlainText(chunk)
        self.last_response += chunk
        
        # Ensure the cursor is at the end and scroll to it
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.response_area.setTextCursor(cursor)
        
        scrollbar = self.response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Force a GUI update
        QApplication.processEvents()


        


    def handle_truncated(self):
        self.response_area.appendPlainText("\n\n[FIGYELMEZTETÉS: A válasz hossza elérte a token limitet!]")
        self.status_bar.showMessage("Figyelem: válasz csonkolt!")
        self.continue_button.setEnabled(True)


    def handle_stream_interrupted(self):
        self.response_area.appendPlainText("\n\n[STREAM MEGSZAKADT - NYOMD A FOLYTATÁS GOMBOT]")
        self.status_bar.showMessage("Stream megszakadt - folytathatod a választ")
        self.continue_button.setEnabled(True)


    def handle_finished(self):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
         
        
        if self.worker.was_truncated:
            self.continue_button.setEnabled(True)
        else:
            self.continue_button.setEnabled(False)
        
        # Kódrészletek kinyerése és megjelenítése
        self.extract_code_blocks()
        self.display_code_blocks()
        self.highlight_code_blocks()
        
        # Beszélgetési történet frissítése
        if not self.is_continuation:
            self.conversation_history.append({"role": "assistant", "content": self.last_response})
        else:
            # Távolítsuk el a "Folytasd" utasítást
            self.conversation_history.pop()
            # De a választ hozzáadjuk a történethez
            self.conversation_history.append({"role": "assistant", "content": self.last_response})
        
        # Automatikus mentés
        if self.current_history_file:
            success, _ = HistoryManager.save_history(self.conversation_history, self.current_history_file)
            if not success:
                self.status_bar.showMessage("Hiba történt az előzmény mentésekor!")
        else:
            success, filename = HistoryManager.save_history(self.conversation_history)
            if success:
                self.current_history_file = filename
                self.status_bar.showMessage(f"Beszélgetés mentve: {filename}")
            else:
                self.status_bar.showMessage("Hiba történt az előzmény mentésekor!")
        
        self.worker = None
        self.last_response = ""
        self.is_continuation = False  # Reset continuation flag
        self.update_history_menu()

    def handle_chunk(self, chunk):
        if not self.worker.response_started:
            # Töröljük a prompt szövegét amikor megérkezik az első válasz
            self.input_field.clear()
            
            model_name = self.worker.model_used
            cursor = self.response_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            # Create format for AI name
            ai_format = QTextCharFormat()
            ai_format.setForeground(QColor("#2ecc71"))  # Light green
            ai_format.setFontWeight(QFont.Bold)
            
            # Insert AI name
            if self.response_area.toPlainText():
                cursor.insertText("\n\n")
            cursor.insertText(f"Szita AI ({model_name}):\n", ai_format)
            
            # Reset to default format for response text
            default_format = QTextCharFormat()
            default_format.setForeground(QColor("#D4D4D4"))
            cursor.setCharFormat(default_format)
            
            self.response_area.setTextCursor(cursor)
            self.worker.response_started = True
        
         
        # Insert the actual response chunks
        self.response_area.insertPlainText(chunk)
        self.last_response += chunk
        
        # Ensure the cursor is at the end and scroll to it
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.response_area.setTextCursor(cursor)
        
        scrollbar = self.response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Force a GUI update
        QApplication.processEvents()

    def save_conversation(self):
        if self.current_history_file:
            success, _ = HistoryManager.save_history(self.conversation_history, self.current_history_file)
            if success:
                self.status_bar.showMessage(f"Beszélgetés frissítve: {self.current_history_file}")
            else:
                QMessageBox.critical(self, "Hiba", "Hiba történt a mentés során!")
        else:
            self.save_conversation_as()


    def extract_code_blocks(self):
        self.current_code_blocks = []
        if not self.worker:
            return
            
        # Javított regex több nyelv és üres sor kezelésére
        pattern = r'```([\w\+]*)\n?([\s\S]*?)\n```'
        matches = re.findall(pattern, self.worker.full_response)
        
        for i, match in enumerate(matches, 1):
            language = match[0].strip() if match[0].strip() else "text"
            code = match[1].strip()
            if code:  # Csak nem üres kódrészleteket adjuk hozzá
                self.current_code_blocks.append({
                    "id": i,
                    "code": code,
                    "language": language
                })


    def highlight_code_blocks(self):
        text = self.response_area.toPlainText()
        cursor = self.response_area.textCursor()
        format = QTextCharFormat()
        format.setBackground(QColor("#1E3A1E"))
        format.setForeground(QBrush(QColor("#29B829")))
        
        for block in self.current_code_blocks:
            # Search for the original code block with backticks
            code_pattern = re.compile(rf'```{block["language"]}\n{re.escape(block["code"])}\n```')
            match = code_pattern.search(text)
            
            if match:
                start_index = match.start()
                end_index = match.end()
                
                cursor.setPosition(start_index)
                cursor.setPosition(end_index, QTextCursor.KeepAnchor)
                cursor.mergeCharFormat(format)


    def display_code_blocks(self):
        # Clear previous code blocks
        for i in reversed(range(self.code_layout.count())): 
            widget = self.code_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Add new code blocks
        for block in self.current_code_blocks:
            widget = CodeBlockWidget(block["code"], block["language"], block["id"])
            self.code_layout.addWidget(widget)


    def handle_error(self, error_msg, error_code):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
         
        self.status_bar.showMessage(f"Hiba: {error_msg}")
        if error_code == 429:
            self.response_area.appendPlainText(f"[HIBA: {error_msg}]")
            QMessageBox.warning(
                self, 
                "Túl sok kérés", 
                "Elérted az API korlátot (5 kérés/perc). Kérlek várj egy percet a következő kérés előtt."
            )
        else:
            self.response_area.appendPlainText(f"[HIBA: {error_msg}]")
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.is_continuation = False  # Reset continuation flag


    def show_context_menu(self, position):
        menu = QMenu(self)
        copy_action = menu.addAction("Kijelölés másolása")
        copy_action.triggered.connect(self.copy_selected)
        
        copy_all_action = menu.addAction("Összes másolása")
        copy_all_action.triggered.connect(self.copy_all)
        
        clear_action = menu.addAction("Törlés")
        clear_action.triggered.connect(self.clear_response)
        
        menu.exec_(self.response_area.viewport().mapToGlobal(position))


    def copy_selected(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.response_area.textCursor().selectedText())
        self.status_bar.showMessage("Kijelölt szöveg másolva!")


    def copy_all(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.response_area.toPlainText())
        self.status_bar.showMessage("Összes szöveg másolva a vágólapra!")


    def clear_response(self):
        self.response_area.clear()
        # Clear code blocks
        for i in reversed(range(self.code_layout.count())): 
            widget = self.code_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.current_code_blocks = []
        self.conversation_history = []
        self.last_response = ""
        self.status_bar.showMessage("Választerület törölve")
        self.continue_button.setEnabled(False)
        self.first_message = True  # Reset system message
        self.current_history_file = None
        self.update_history_menu()

 

    def save_conversation_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Beszélgetés mentése", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                # Ensure it has .json extension
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # Save to history directory
                filename = os.path.basename(file_path)
                success, saved_path = HistoryManager.save_history(self.conversation_history, filename)
                
                if success:
                    self.current_history_file = filename
                    self.status_bar.showMessage(f"Beszélgetés mentve: {filename}")
                    self.update_history_menu()
                else:
                    QMessageBox.critical(self, "Hiba", f"Fájl mentési hiba: {saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hiba", f"Fájl mentési hiba: {str(e)}")


    def load_conversation(self, filename=None):
        if not filename:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Beszélgetés betöltése", HISTORY_DIR, "JSON Files (*.json);;All Files (*)"
            )
            if not file_path:
                return
            filename = os.path.basename(file_path)
        
        success, history = HistoryManager.load_history(filename)
        if success:
            self.conversation_history = history
            self.current_history_file = filename
            
            # Reconstruct the conversation in the response area
            self.response_area.clear()
            for msg in history:
                if msg["role"] == "user":
                    self.response_area.appendPlainText(f"Felhasználó: {msg['content']}")
                elif msg["role"] == "assistant":
                    self.response_area.appendPlainText(f"Szita AI: {msg['content']}")
                elif msg["role"] == "system" and msg['content'] != SYSTEM_MESSAGE['content']:
                    self.response_area.appendPlainText(f"[Rendszer]: {msg['content']}")
            
            # Extract and display code blocks
            full_text = self.response_area.toPlainText()
            self.extract_code_blocks_from_text(full_text)
            self.display_code_blocks()
            self.highlight_code_blocks()
            
            self.status_bar.showMessage(f"Beszélgetés betöltve: {filename}")
            self.first_message = False
            self.update_history_menu()
        else:
            QMessageBox.critical(self, "Hiba", f"Fájl betöltési hiba: {history}")


    def extract_code_blocks_from_text(self, text):
        self.current_code_blocks = []
        pattern = r'```([\w\+]*)\n?([\s\S]*?)\n```'
        matches = re.findall(pattern, text)
        
        for i, match in enumerate(matches, 1):
            language = match[0].strip() if match[0].strip() else "text"
            code = match[1].strip()
            if code:
                self.current_code_blocks.append({
                    "id": i,
                    "code": code,
                    "language": language
                })


    def update_history_menu(self):
        self.history_menu.clear()
        
        # Add "Clear History" action
        clear_action = QAction("Előzmények törlése", self)
        clear_action.triggered.connect(self.clear_history)
        self.history_menu.addAction(clear_action)
        self.history_menu.addSeparator()
        
        # Add history files
        history_files = HistoryManager.list_history_files()
        for file in history_files:
            action = QAction(file, self)
            action.triggered.connect(lambda checked, f=file: self.load_conversation(f))
            self.history_menu.addAction(action)


    def clear_history(self):
        reply = QMessageBox.question(
            self, 
            "Előzmények törlése", 
            "Biztosan törölni akarod az összes előzményfájlt?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for file in HistoryManager.list_history_files():
                HistoryManager.delete_history_file(file)
            self.status_bar.showMessage("Minden előzmény törölve!")
            self.update_history_menu()


    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self.save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    #  --hidden-import psutil 