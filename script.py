import importlib, sys
try:
    import importlib
    import energy
    importlib.reload(energy)
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_ERROR', type(e).__name__, e)
