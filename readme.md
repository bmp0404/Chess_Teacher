
# Chess Engine Setup Guide




----------

## Installing Required Packages

Before running the engine, make sure to install all the necessary Python packages:

```bash
pip install -r requirements.txt

```

----------

## Patching CairoSVG on Windows

If you encounter errors related to **CairoSVG**, it's because the Cairo graphics library is missing. Follow these steps to fix the issue.

### Step 1: Install GTK3 (Includes Cairo)

1.  **Download GTK3 Runtime:**
    
    -   Visit [GTK for Windows](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
    -   Download the latest GTK3 Runtime installer for **Win64**.
2.  **Install GTK3:**
    
    -   Run the downloaded `.exe` file.
    -   Make sure to check the option **"Set up PATH environment variable"** during installation.

----------

### Step 2: Verify Cairo Installation

-   Open **File Explorer** and navigate to:
    
    ```
    C:\Program Files\GTK3-Runtime Win64\bin
    
    ```
    
-   Ensure the file **`libcairo-2.dll`** is present.

----------

### Step 3: Add GTK to System PATH (If Not Added Automatically)

If the GTK installer didn’t add the path automatically:

1.  Press **Win + R**, type `sysdm.cpl`, and press **Enter**.
2.  Go to **Advanced** → **Environment Variables**.
3.  Under **System Variables**, find `Path` and click **Edit**.
4.  Click **New** and add:
    
    ```
    C:\Program Files\GTK3-Runtime Win64\bin
    
    ```
    
5.  Click **OK** and restart your computer.

----------

### Step 4: Reinstall CairoSVG and Dependencies

1.  **Uninstall existing packages:**
    
    ```bash
    pip uninstall cairosvg cairocffi pycairo
    
    ```
    
2.  **Reinstall CairoSVG:**
    
    ```bash
    pip install cairosvg
    
    ```
    
3.  **Ensure PyCairo is installed:**
    
    ```bash
    pip install pycairo
    
    ```
    

----------

### Step 5: Add Python Scripts Directory to PATH (If Needed)

To avoid potential errors, manually add the Python scripts directory to your system’s PATH.

1.  **Open System Environment Variables:**
    
    -   Press **Win + R**, type `sysdm.cpl`, and press **Enter**.
    -   Go to the **Advanced** tab and click **Environment Variables**.
2.  **Edit the PATH Variable:**
    
    -   Under **System Variables**, find `Path` and click **Edit**.
    -   Click **New** and add:
        
        ```
        C:\Users\your_user\AppData\Roaming\Python\Python312\Scripts
        
        ```
        
    -   Click **OK** to save the changes.

## Running the Chess Engine

These commands allow you to run a simple game where both opponents play random moves, just to verify the engine is functional.

### For Windows

```bash
python engine_driver.py

```

### For Linux/Mac

```bash
python3 engine_driver.py

```