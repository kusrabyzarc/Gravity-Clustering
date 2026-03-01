@echo off
REM Активируем виртуальное окружение
call .venv\Scripts\activate.bat

REM Запуск Streamlit
streamlit run app.py
pause
