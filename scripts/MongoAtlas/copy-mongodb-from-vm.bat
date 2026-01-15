@echo off
REM Copy MongoDB data from VM (EWRSPT-AI) to local Docker MongoDB
REM Usage: copy-mongodb-from-vm.bat [local_container_name]

setlocal

REM Configuration
set VM_HOST=chad.walker@EWRSPT-AI
set VM_CONTAINER=local27018
set VM_BACKUP_PATH=C:/Projects/mongo-backup
set LOCAL_BACKUP_PATH=C:\Projects\mongo-backup
set LOCAL_CONTAINER=local27019

REM Override with command line arg if provided
if not "%~1"=="" set LOCAL_CONTAINER=%~1

echo.
echo ========================================
echo MongoDB VM to Local Copy Script
echo ========================================
echo VM Host: EWRSPT-AI
echo VM Container: %VM_CONTAINER%
echo Local Container: %LOCAL_CONTAINER%
echo ========================================
echo.

REM Step 1: Create archive dump on VM
echo [1/3] Creating MongoDB archive on VM...
ssh %VM_HOST% "docker exec %VM_CONTAINER% mongodump --archive=/tmp/mongo.archive"
if errorlevel 1 (
    echo ERROR: Failed to create dump on VM
    goto :error
)

REM Step 2: Copy archive from container to VM, then to local
echo [2/3] Copying archive to local PC...
if not exist "%LOCAL_BACKUP_PATH%" mkdir "%LOCAL_BACKUP_PATH%"
ssh %VM_HOST% "docker cp %VM_CONTAINER%:/tmp/mongo.archive C:/Projects/mongo.archive"
if errorlevel 1 (
    echo ERROR: Failed to copy archive from container
    goto :error
)

copy "\\EWRSPT-AI\C$\Projects\mongo.archive" "%LOCAL_BACKUP_PATH%\mongo.archive" /Y
if errorlevel 1 (
    echo ERROR: Failed to copy archive to local PC
    goto :error
)

REM Step 3: Restore locally
echo [3/3] Restoring to local MongoDB container...
docker cp %LOCAL_BACKUP_PATH%\mongo.archive %LOCAL_CONTAINER%:/tmp/mongo.archive
if errorlevel 1 (
    echo ERROR: Failed to copy archive into container
    goto :error
)

docker exec %LOCAL_CONTAINER% mongorestore --archive=/tmp/mongo.archive --drop --noIndexRestore
if errorlevel 1 (
    echo ERROR: Failed to restore MongoDB
    goto :error
)

echo.
echo ========================================
echo SUCCESS: MongoDB data copied from VM to local
echo ========================================
echo.
goto :end

:error
echo.
echo ========================================
echo FAILED: See error messages above
echo ========================================
echo.
exit /b 1

:end
REM Cleanup: Remove archive files
echo Cleaning up temporary files...
ssh %VM_HOST% "docker exec %VM_CONTAINER% rm -f /tmp/mongo.archive"
ssh %VM_HOST% "del C:\Projects\mongo.archive 2>nul"
docker exec %LOCAL_CONTAINER% rm -f /tmp/mongo.archive

echo Done!
endlocal
