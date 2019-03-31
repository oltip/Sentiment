from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO


from fastai.text import * 

# Here give where is the file / google cloud 
export_file_url = 'https://drive.google.com/file/d/1BQTagMmaZeY0zSWp1CppRrHlM--WviTK/view'


# we have created another folder within the datafile in order to use the same code 
#model_file_name = 'model'    # actually, if you click on order, it will create another url 
export_file_name = 'export.pkl'  


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
    await download_file(export_file_url, path/export_file_name)    # Do we need to download the encoder as well??? 

    learn = load_learner(path, export_file_name)
    return learn 


# This piece should be standard 
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index_1.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.json()
    print('data:', data ) 
	
    img = data['textField']            # ok, it should be a part of Json, a textField key value 
#   What we have done in the previous line of code, is to receive the text that people would insert 


    print("data['textField']", data["textField"])
    print("img:", img)

#    prediction = learn.predict(img)[0]
    prediction = learn.predict(img)     # show all of that string object

    # THIS IS THE RESULT THAT WILL BE SHOWN 

    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
