# Brain tumor classification model running dashboard guide
## Setup Virtual Environment


1. **Setup Environment Anaconda**:
    ```bash
    conda create --name main-ds python=3.9
    conda activate main-ds
    pip install -r requirements.txt
    ```

2. **Setup Environment - Shell/Terminal**:
    ```bash
    mkdir proyek_analisis_data_bike_sharing
    cd proyek_analisis_data_bike_sharing
    pipenv install
    pipenv shell
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run main.py
    ```