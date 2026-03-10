@echo off
echo 清理系统临时目录中的 temp_extract_* 文件夹...
echo.

:: 设置临时目录路径
set TEMP_DIR=%USERPROFILE%\AppData\Local\Temp

:: 查找并删除所有 temp_extract_ 开头的文件夹
echo 正在删除 %TEMP_DIR%\temp_extract_* 文件夹...
for /d %%d in ("%TEMP_DIR%\temp_extract_*") do (
    echo 删除: %%d
    rd /s /q "%%d"
)

echo.
echo 清理完成！
pause