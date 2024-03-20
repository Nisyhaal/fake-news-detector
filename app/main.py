import os
import sys
import uvicorn

sys.path.append(os.getcwd())
import config

if __name__ == "__main__":
    from detector_api import app
    uvicorn.run("detector_api:app", host=config.APP_HOST, port=config.APP_PORT, reload=config.APP_AUTO_RELOAD)
