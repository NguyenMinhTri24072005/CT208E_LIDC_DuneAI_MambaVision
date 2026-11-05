@echo off
REM Chuyen thu muc ve dung vi tri cua file bat
cd /d "%~dp0"

REM Thiet lap bang ma UTF-8 de tranh loi tieng Viet / Emoji
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Kich hoat moi truong ao neu co
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo ==============================================================
echo BAT DAU CHANG 1: CHUYEN DOI TOAN BO DICOM SANG NRRD
echo ==============================================================
python data/data_pipeline/dicom_to_nrrd.py
if errorlevel 1 goto error

echo.
echo ==============================================================
echo BAT DAU CHANG 2: TIEN XU LY Y KHOA (NRRD -^> CACHE 1MM)
echo ==============================================================
echo Dang boc tach phoi va chuan hoa 1mm... Tien trinh nay se mat nhieu gio!
python DuneAI_Remake/src/medical_prep.py build --nrrd-dir data/duneai_raw_nrrd --cache-dir data/cache_1mm --workers 2 --extract-lungs
if errorlevel 1 goto error

echo.
echo ==============================================================
echo BAT DAU CHANG 3: TAO THUC DON HUAN LUYEN (SLICES.CSV)
echo ==============================================================
python DuneAI_Remake/src/offline_builder.py build --cache-dir data/cache_1mm
if errorlevel 1 goto error

echo.
echo ==============================================================
echo HOAN TAT TOAN BO TIEN TRINH CAM MAY!
echo ==============================================================
pause
exit /b 0

:error
echo.
echo [!] CO LOI XAY RA TRONG QUA TRINH XU LY!
pause
exit /b 1