#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/stdio.h>
#elif !defined(__linux__)
#error "Invar publication requires Linux or macOS filesystem operations"
#endif

int invar_rename(int directory, const char *source, const char *destination);
int invar_reference(int directory, const char *source, const char *destination);
int invar_sync_file(int file);
int invar_sync_directory(int directory, int file);

int invar_rename(int directory, const char *source, const char *destination) {
#if defined(__APPLE__)
    return renameatx_np(directory, source, directory, destination, RENAME_EXCL);
#else
    return renameat2(directory, source, directory, destination, RENAME_NOREPLACE);
#endif
}

int invar_sync_file(int file) {
#if defined(__APPLE__)
    return fcntl(file, F_FULLFSYNC);
#else
    return fsync(file);
#endif
}

int invar_reference(int directory, const char *source, const char *destination) {
    return symlinkat(source, directory, destination);
}

int invar_sync_directory(int directory, int file) {
#if defined(__APPLE__)
    if (fsync(directory) == -1) {
        return -1;
    }
    return fcntl(file, F_FULLFSYNC);
#else
    (void)file;
    return fsync(directory);
#endif
}
