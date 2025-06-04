import json
import os
import re
import threading
from tkinter import TclError
from .models import Compound
from .preprocessing import tic_scaling
import customtkinter as ctk


def parse_core(file_path=None, load_cache=True, cache_file='library/NIST14.json', subset=None, 
               progress_callback=None, save_path=None):
    """
    Core parsing logic without any UI dependencies.
    
    Args:
        file_path: Path to the text file to parse
        load_cache: Whether to try loading from cache first
        cache_file: Path to the cache file
        subset: Optional subset of elements to filter by
        progress_callback: Function to call with progress updates (0.0-1.0)
        save_path: Optional path to save the filtered subset to (if different from cache_file)
    
    Returns:
        Dictionary of compounds
    """
    compounds = {}
    
    def report_progress(value):
        """Report progress to callback if provided"""
        if progress_callback:
            try:
                progress_callback(value)
            except Exception:
                pass
    
    try:
        if subset:
            allowed_elements = set(subset.upper())

        if load_cache and os.path.exists(cache_file):
            print('Loading library from JSON file...')
            with open(cache_file, 'r') as json_file:
                data = json.load(json_file)
                total_items = len(data)
                for i, (k, v) in enumerate(data.items(), start=1):
                    # v may be either a mapping (dict) or already a Compound; handle both.
                    compound = v if isinstance(v, Compound) else Compound.from_json(v)
                    if subset:
                        if all(element in allowed_elements for element in re.findall(r'[A-Za-z]', compound.formula)):
                            compounds[k] = compound
                    else:
                        compounds[k] = compound
                    report_progress(i / total_items)
            print('Library successfully loaded from cache.')
            
            # Save subset to separate file if requested and subset filtering was applied
            if save_path and save_path != cache_file and subset:
                print(f'Saving filtered subset to {save_path}...')
                with open(save_path, 'w') as json_file:
                    # Convert compounds to JSON serializable format
                    json_compounds = {k: v.to_json() for k, v in compounds.items()}
                    json.dump(json_compounds, json_file)
                print(f'Subset successfully saved to {save_path}.')
                
        else:
            # Only try to parse text file if file_path is provided
            if file_path is None:
                raise ValueError("Cannot parse library: no text file path provided and cache loading failed or disabled")
                
            with open(file_path, 'r') as file:
                lines = file.readlines()
                total_lines = len(lines)
                current_compound = None

                for i, line in enumerate(lines, start=1):
                    line = line.strip()
                    if line.startswith("Name:"):
                        if current_compound:
                            current_compound.spectrum = tic_scaling(current_compound.spectrum)
                            compounds[current_compound.name] = current_compound.to_json()
                        current_compound = Compound()
                        current_compound.name = line.split(":", 1)[1].strip()
                    elif line.startswith("Formula:"):
                        current_compound.formula = line.split(":", 1)[1].strip()
                    elif line.startswith("MW:"):
                        current_compound.mw = float(line.split(":", 1)[1].strip())
                    elif line.startswith("CASNO:"):
                        casno = line.split(":", 1)[1].strip()
                        casno = ''.join(filter(str.isdigit, casno))
                        if len(casno) == 9:
                            formatted_casno = f"{casno[:3]}-{casno[3:5]}-{casno[5:]}"
                            current_compound.casno = formatted_casno
                        else:
                            current_compound.casno = casno
                    elif line.startswith("ID:"):
                        current_compound.id_ = int(line.split(":", 1)[1].strip())
                    elif line.startswith("Comment:"):
                        current_compound.comment = line.split(":", 1)[1].strip()
                    elif line.startswith("Num peaks:"):
                        current_compound.num_peaks = int(line.split(":", 1)[1].strip())
                    elif current_compound.num_peaks is not None:
                        if current_compound.spectrum is None:
                            current_compound.spectrum = []
                        spectrum_data = line.split()
                        if len(spectrum_data) == 2:
                            current_compound.spectrum.append((int(spectrum_data[0]), float(spectrum_data[1])))
                    report_progress(i / total_lines)

                if current_compound:
                    current_compound.spectrum = tic_scaling(current_compound.spectrum)
                    compounds[current_compound.name] = current_compound.to_json()

                # Save to specified cache file
                with open(cache_file, 'w') as json_file:
                    json.dump(compounds, json_file)
                    print('JSON file successfully created.')
                
                # If subset filtering was applied, save to separate file if requested
                if save_path and save_path != cache_file and subset:
                    # Filter by subset before saving
                    filtered_compounds = {}
                    for k, v in compounds.items():
                        compound = v if isinstance(v, Compound) else Compound.from_json(v)
                        if all(element in allowed_elements for element in re.findall(r'[A-Za-z]', compound.formula)):
                            filtered_compounds[k] = v
                    
                    with open(save_path, 'w') as json_file:
                        json.dump(filtered_compounds, json_file)
                        print(f'Filtered subset successfully saved to {save_path}.')

        # Convert stored JSON data back to Compound instances if needed.
        result = {k: v if isinstance(v, Compound) else Compound.from_json(v)
                  for k, v in compounds.items()}
        report_progress(1.0)
        return result
    except Exception as e:
        print(f"Error during parsing: {str(e)}")
        report_progress(1.0)  # Ensure we complete the progress indication
        raise


def parse(file_path=None, load_cache=True, cache_file='library/NIST14.json', subset=None, 
          show_ui=True, ui_framework='ctk', progress_callback=None, save_path=None):
    """
    Parse MS library files with flexible UI options.
    
    Args:
        file_path: Path to the text file to parse
        load_cache: Whether to try loading from cache first
        cache_file: Path to the cache file
        subset: Optional subset of elements to filter by
        show_ui: Whether to show a progress UI (default: True)
        ui_framework: 'ctk' for CustomTkinter or 'pyside6' for PySide6
        progress_callback: Custom progress callback function
        save_path: Optional path to save the filtered subset to (if different from cache_file)
        
    Returns:
        Dictionary of compounds
    """
    # If we have a custom callback or don't want UI, just call parse_core directly
    if progress_callback or not show_ui:
        return parse_core(
            file_path=file_path, 
            load_cache=load_cache, 
            cache_file=cache_file, 
            subset=subset,
            progress_callback=progress_callback,
            save_path=save_path
        )
    
    # Otherwise set up the appropriate UI
    if ui_framework.lower() == 'ctk':
        return _parse_with_ctk(file_path, load_cache, cache_file, subset, save_path)
    elif ui_framework.lower() == 'pyside6':
        return _parse_with_pyside6(file_path, load_cache, cache_file, subset, save_path)
    else:
        raise ValueError(f"Unknown UI framework: {ui_framework}. Use 'ctk' or 'pyside6'.")

def _parse_with_ctk(file_path, load_cache, cache_file, subset, save_path=None):
    """Implementation with CustomTkinter UI"""
    import customtkinter as ctk
    from tkinter import TclError
    
    result_container = {}
    progress_value = [0]
    
    def parse_thread_function():
        try:
            def update_progress(value):
                progress_value[0] = value
                
            result = parse_core(
                file_path=file_path,
                load_cache=load_cache,
                cache_file=cache_file,
                subset=subset,
                progress_callback=update_progress,
                save_path=save_path
            )
            result_container['result'] = result
        finally:
            progress_value[0] = 1.0
    
    # Create a dedicated root window for the loading screen
    loading_window = ctk.CTk()
    loading_window.title("Parsing File")
    window_width = 300
    window_height = 100
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    position_x = (screen_width // 2) - (window_width // 2)
    position_y = (screen_height // 2) - (window_height // 2)
    loading_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    progress_var = ctk.DoubleVar(value=0)
    progress_bar = ctk.CTkProgressBar(loading_window, variable=progress_var, progress_color='green')
    progress_bar.pack(pady=10, padx=20, fill="x")
    progress_label = ctk.CTkLabel(loading_window, text="Loading MS Library...")
    progress_label.pack(pady=10)

    # Start the parsing thread
    parsing_thread = threading.Thread(target=parse_thread_function)
    parsing_thread.daemon = True
    parsing_thread.start()

    def update_progress_ui():
        try:
            if not loading_window.winfo_exists():
                return
            
            # Update the progress bar with the current value
            progress_var.set(progress_value[0])
            
            # Schedule the next update if thread is still running
            if parsing_thread.is_alive():
                loading_window.after(100, update_progress_ui)
            else:
                # Thread is done, set to 100% and close
                progress_var.set(1.0)
                loading_window.after(500, loading_window.destroy)
        except TclError:
            pass

    # Start the update loop and enter Tk mainloop
    loading_window.after(100, update_progress_ui)
    loading_window.mainloop()

    return result_container.get('result', {})

def _parse_with_pyside6(file_path, load_cache, cache_file, subset, save_path=None):
    """Implementation with PySide6 UI"""
    try:
        from PySide6.QtWidgets import (QApplication, QDialog, QProgressBar, 
                                      QVBoxLayout, QLabel, QFrame, QHBoxLayout)
        from PySide6.QtCore import Qt, QSize
        from PySide6.QtGui import QIcon, QPixmap
        import os
    except ImportError:
        raise ImportError("PySide6 is required for PySide6 UI. Install with 'pip install PySide6'")
    
    # Create or get QApplication
    app = QApplication.instance() or QApplication([])
    
    # Create a custom dialog for a more modern look
    dialog = QDialog()
    dialog.setWindowTitle("ms-toolkit")
    dialog.setFixedSize(350, 140)  # Slightly taller to accommodate the header
    dialog.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
    
    # Load the icon for both window icon and display
    icon_path = os.path.join('resources', 'icon.png')
    if os.path.exists(icon_path):
        dialog.setWindowIcon(QIcon(icon_path))
    
    # Create main layout
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 15, 20, 15)
    layout.setSpacing(10)
    
    # Create header with icon and title
    header_layout = QHBoxLayout()
    header_layout.setSpacing(10)
    
    # Add icon to the header if it exists
    if os.path.exists(icon_path):
        icon_label = QLabel()
        pixmap = QPixmap(icon_path)
        icon_label.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_layout.addWidget(icon_label)
    
    # Add title to header
    title_label = QLabel("Loading MS Library...")
    title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    header_layout.addWidget(title_label)
    header_layout.addStretch()
    
    # Add the header layout to the main layout
    layout.addLayout(header_layout)
    
    # Add a separator line
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    separator.setStyleSheet("background-color: #E0E0E0; margin: 5px 0px;")
    layout.addWidget(separator)
    
    # Create a modern progress bar
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_bar.setValue(0)
    progress_bar.setTextVisible(True)
    progress_bar.setFormat("%p%")
    progress_bar.setStyleSheet("""
        QProgressBar {
            border: 1px solid #E0E0E0;
            border-radius: 5px;
            background-color: #F5F5F5;
            text-align: center;
            height: 20px;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 5px;
        }
    """)
    layout.addWidget(progress_bar)
    
    # Add status label below progress bar
    status_label = QLabel("Initializing...")
    status_label.setStyleSheet("color: #707070; font-size: 11px;")
    layout.addWidget(status_label)
    
    dialog.show()
    
    # Function to update progress
    def update_progress(value):
        percent = int(value * 100)
        progress_bar.setValue(percent)
        
        # Update status message based on progress
        if value < 0.01:
            status_label.setText("Initializing...")
        elif value < 0.5:
            status_label.setText("Loading library data...")
        elif value < 0.9:
            status_label.setText("Processing compounds...")
        else:
            status_label.setText("Finalizing...")
            
        QApplication.processEvents()
    
    # Parse with the progress callback
    result = parse_core(
        file_path=file_path,
        load_cache=load_cache,
        cache_file=cache_file,
        subset=subset,
        progress_callback=update_progress,
        save_path=save_path
    )
    
    # Mark as complete and wait briefly before closing
    progress_bar.setValue(100)
    status_label.setText("Complete!")
    QApplication.processEvents()
    
    # Use QTimer for a short delay before closing
    from PySide6.QtCore import QTimer
    QTimer.singleShot(800, dialog.close)
    
    # If the application was created here, make sure it processes remaining events
    if not QApplication.instance().startingUp():
        app.processEvents()
    
    return result
