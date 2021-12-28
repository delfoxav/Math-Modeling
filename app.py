from mimetypes import init
import dash
from dash.html.A import A
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import time as time
from dash.exceptions import PreventUpdate

i=1

#Function to define if a number is a perfect square:
def isPerfectsquare(number):
    if np.sqrt(number)%1==0:
        return True

#Function to create initial square distribution:
def CreateSquare(initA,initB):
    img_rgb=[1]*initA+[0]*initB
    np.random.shuffle(img_rgb)
    img_rgb=np.reshape(np.array(img_rgb),(int(np.sqrt(initA+initB)),int(np.sqrt(initA+initB))))
    img_rgb=np.array(img_rgb,dtype=np.uint8)
    #img_rgb=nparray.tolist()

    return img_rgb

def CalculateNextStep(A,B):
    A+=1
    B-=1
    return (A,B)

fig=px.imshow([[0,0],
              [0,0]],binary_string=True)

app = dash.Dash(__name__)

img_rgb = [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
           [[0, 255, 0], [0, 0, 255], [255, 0, 0]]]

app.layout = html.Div([
    html.H1("Please define the maximum size of the population, the initial distribution and the fitness"),
    html.Div([
        "Initial A individuals: ",
        dcc.Input(id='A',
                  value=100,
                  type='number',
                  min = 1,
                  step = 1),
        "Initial B individuals: ",
        dcc.Input(id='B',
                  value=100,
                  type='number',
                  min = 1,
                  step = 1),
        "fitness: ",
        dcc.Input(id='fitness',
                  value=1,
                  type='number',
                  min = 1,
                  step = 1),
        "Steps: ",
        dcc.Input(id='Step',
                  value=10,
                  type='number',
                  min=1,
                  step=1)
        
    ]),
    html.Div([
        "fitness on: ",
        dcc.RadioItems(id='negfit',
                       options=[{'label': i, 'value': i} for i in ['A', 'B']],     
                       value='A',
                        labelStyle={'display': 'inline-block'}
                        ),
        html.Button('Start',id='submit-val',n_clicks=0),
        html.Button('Stop',id='stop-val',n_clicks=0),
        html.Div(id='slider-output-distribution', style={'margin-top': 20})
    ]),
    html.Div([dcc.Graph(id="graph",figure=fig)]),
    dcc.Interval(id="refresh-graph-interval", disabled=True, interval=1*500, n_intervals=10)
])

#Store the number of step NOT WORKING YET
@app.callback(
    Output("refresh-graph-interval","n_intervals"),
    Input("Step","value")
)
def update_interval(step):
    return step


#Rebuild the plot according to the value of A and B
@app.callback(
    Output(component_id='slider-output-distribution',component_property='children'),
    Output("refresh-graph-interval","disabled"),
    State('A', 'value'),
    State("B","value"),
    Input('submit-val','n_clicks'),
    Input('stop-val','n_clicks'))

def Simulate(A,B,submit,stop):
    ctx = dash.callback_context
    button = ctx.triggered[0]['prop_id'].split('.')[0]
    print(button)
    if button == 'stop-val':
        global i
        i=0
        return "", True
    elif button == "submit-val":
        Z = [True]*A+[False]*B
        np.random.shuffle(Z)
        #Z = np. reshape(np.array(Z), (np.sqrt(init_A+init_B), np.sqrt(init_A+init_B)))
        if not isPerfectsquare(B+A):
            return "Please change the numbers to form a square",True
        else:
        
            return "", False

   
@app.callback(
    Output('graph','figure'),
    Output('A','value'),
    Output('B','value'),
    Input("refresh-graph-interval","n_intervals"),
    State('A', 'value'),
    State("B","value")
    
    )
def update_graph(n,A,B):
    global i
    i=i+1
    if (A==0):
        fig=px.imshow([[1,1],[1,1]])
    elif(B==0):
        fig=px.imshow([[0,0],[0,0]])
        
    else:
        A,B=CalculateNextStep(A,B)
        fig=px.imshow(CreateSquare(A,B))
        fig.update_layout(title_text="Step "+str(i),
            title_font_size=30)
        time.sleep(0.1)
    return fig,A,B
    



if __name__ == '__main__':
    app.run_server(debug=True)