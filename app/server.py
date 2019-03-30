from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
# from fastai.vision import *
from fastai.text import * 

# Here give where is the file / google cloud 
model_file_url = 'https://drive.google.com/drive/u/0/folders/1oqKpXXdxn3_IhVY2A779wNZB2BX8iDPu'

# we have created another folder within the datafile in order to use the same code 
model_file_name = 'model'    # actually, if you click on order, it will create another url 


classes = ['positive', 'negative'] 

path = Path(__file__).parent    # hmmm.... What is this??? 

app = Starlette() 

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])  # ??? 
app.mount('/static', StaticFiles(directory='app/static'))   # this has to do with rendering 


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data) 


async def setup_learner():
	# Ok, here we have to intervene 
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')    # Do we need to download the encoder as well??? 

    # change Image to Text and modify the rest 
    data_bunch = TextDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)

    # This should be adapted for text 
    learn = create_cnn(data_bunch, models....., pretrained=False)    # Why pretrained False? 
    learn.load(model_file_name)
    return learn

# This piece should be standard 
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
