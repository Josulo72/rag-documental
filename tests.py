import ast,sys,json,math,io,hashlib,os,tempfile
from types import SimpleNamespace as NS
from pypdf import PdfWriter,PdfReader
from pathlib import Path
source=Path(__file__).with_name('app.py').read_text(encoding='utf-8')
tree=ast.parse(source, filename='app.py')
selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in {'clean_topic','accepted_hits','validate_answer','pdf_documents','demo_documents','load_env','api_settings'}]
class Document:
    def __init__(self,text,metadata,**kwargs):
        self.text=text
        self.metadata=metadata
env={'json':json,'math':math,'io':io,'hashlib':hashlib,'PdfReader':PdfReader,'Document':Document,'NO_ANSWER':'SIN_DATOS','os':os,'Path':Path,'ROOT':Path(__file__).parent,'API_URL':'https://ejemplo.invalid/v1'}
exec(compile(ast.Module(body=selected,type_ignores=[]),'app.py','exec'),env)
count=0
def check(condition):
    global count
    assert condition
    count+=1
def rejects(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False
def hit(topic,score):
    return NS(node=NS(metadata={'tema':topic}),score=score)
check(env['clean_topic']('  Compras  y   ventas ')=='Compras y ventas')
check(rejects(env['clean_topic'],' '))
check(rejects(env['clean_topic'],'a'*81))
values=[hit('Compras',.9),hit('Personal',1),hit('Compras',float('nan')),hit('Compras',None),hit('Compras',.6),hit('Compras',float('inf'))]
out=env['accepted_hits'](values,'Compras',.7)
check(len(out)==1 and out[0].score==.9)
check(len(env['accepted_hits']([hit('Compras',.7)],'Compras',.7))==1)
check(env['accepted_hits'](values,'Ausente',0)==[])
check(env['validate_answer']({'answer':'texto','sources':[2,1,2]},2)==('texto',[2,1]))
check(env['validate_answer']({'answer':'','sources':[]},2)==('SIN_DATOS',[]))
for payload in [None,[],{'answer':'texto','sources':[]},{'answer':'texto','sources':[0]},{'answer':'texto','sources':[3]},{'answer':'texto','sources':[True]},{'answer':2,'sources':[1]}]:
    check(rejects(env['validate_answer'],payload,2))
demo=env['demo_documents']()
check(len(demo)==4)
check({d.metadata['tema'] for d in demo}=={'Compras','Personal'})
class Upload:
    def __init__(self,raw,name):
        self.raw=raw; self.name=name
    def getvalue(self):
        return self.raw
writer=PdfWriter(); writer.add_blank_page(width=100,height=100)
buffer=io.BytesIO(); writer.write(buffer)
check(rejects(env['pdf_documents'],[Upload(buffer.getvalue(),'escaneado.pdf')],['Compras']))
writer.encrypt('clave')
buffer=io.BytesIO(); writer.write(buffer)
check(rejects(env['pdf_documents'],[Upload(buffer.getvalue(),'protegido.pdf')],['Compras']))
saved={k:os.environ.pop(k,None) for k in ('OPENAI_API_KEY','OPENAI_BASE_URL','PRUEBA_RAG')}
check(rejects(env['api_settings']))
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'.env'
    path.write_text('# nota\nOPENAI_API_KEY="clave-de-prueba"\nOPENAI_BASE_URL=\nPRUEBA_RAG=uno\n',encoding='utf-8')
    os.environ['PRUEBA_RAG']='original'
    env['load_env'](path)
    env['load_env'](Path(folder)/'no-existe')
check(os.environ['PRUEBA_RAG']=='original')
check(env['api_settings']()==('clave-de-prueba','https://ejemplo.invalid/v1'))
os.environ['OPENAI_BASE_URL']='http://127.0.0.1:9/v1/'
check(env['api_settings']()[1]=='http://127.0.0.1:9/v1')
for k,v in saved.items():
    os.environ.pop(k,None)
    if v is not None: os.environ[k]=v
print('Sintaxis Python correcta.')
print(str(count)+' comprobaciones correctas: temas, umbrales, referencias, ejemplos, PDF sin texto, PDF protegido y configuracion de la API.')
print('Estas comprobaciones no arrancan Streamlit ni LlamaIndex ni llaman a ningun modelo.')
