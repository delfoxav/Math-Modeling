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
import random

#i=0

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
 
def Updatesquare(previousSquare,toBirth,toDie,A,B):
    previousSquare=np.asarray(previousSquare)
    if toDie == 1 and toBirth == 0:
        x, y = np.where(previousSquare == 1)
        if len(x) > 0:
            i = np.random.randint(len(x))
            random_pos=(x[i], y[i])
            previousSquare[random_pos] = 0
            
    if toDie == 0 and toBirth == 1:
        x, y = np.where(previousSquare == 0)
        if len(x) > 0:
            i = np.random.randint(len(x))
            random_pos=(x[i],y[i])
            previousSquare[random_pos] = 1
    previousSquare =np.reshape(np.array(previousSquare),(int(np.sqrt(A+B)),int(np.sqrt(A+B))))
    return previousSquare

 
def CalculateNextStep(A,B,fitness,selection_on):
    if selection_on=="A":
        birththreshold=(A*fitness)/((A*fitness)+B)
    elif selection_on =="B":
        birththreshold=(A)/(A+(B*fitness))
    deaththreshold=A/(A+B)
    birth=random.uniform(0,1)
    death=random.uniform(0,1)
    if birth <= birththreshold:
        A+=1
        birth = 1
    else:
        B+=1
        birth = 0
    if death <=deaththreshold:
        A-=1
        death = 1
    else:
        B-=1
        death = 0
    return (A,B, birth, death)

img = [[0,0],[0,0]]

fig=px.imshow(img,binary_string=True)

app = dash.Dash(__name__)

img_rgb = [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
           [[0, 255, 0], [0, 0, 255], [255, 0, 0]]]

server = app.server

app.layout = html.Div([
    html.H1("Please define the initial distribution, the fitness and the number of steps"),
    html.Div([
        "Initial A individuals: ",
        dcc.Input(id='A',
                  value=2,
                  type='number',
                  min = 1,
                  step = 1),
        "Initial B individuals: ",
        dcc.Input(id='B',
                  value=2,
                  type='number',
                  min = 1,
                  step = 1),
        "fitness: ",
        dcc.Input(id='fitness',
                  value=1,
                  type='number',
                  min = 1),
        "Steps: ",
        dcc.Input(id='Step',
                  value=10,
                  type='number',
                  min=1,
                  step=1)
        
    ]),
    html.Div([
        dcc.Store(id='storeSquare',storage_type='local',data= CreateSquare(2,2)),
        html.Br(),
        "fitness on: ",
        dcc.RadioItems(id='negfit',
                       options=[{'label': i, 'value': i} for i in ['A', 'B']],     
                       value='A',
                        labelStyle={'display': 'inline-block'}
                        ),
        html.Br(),
        html.Button('Start',id='submit-val',n_clicks=0),
        html.Button('Stop',id='stop-val',n_clicks=0),
        html.Div(id='slider-output-distribution', style={'margin-top': 20})
    ]),
    html.Div([dcc.Graph(id="graph",figure=fig)]),
    dcc.Interval(id="refresh-graph-interval", disabled=True, interval=1*500, n_intervals=0),
    html.Div([
         dcc.Textarea(
        id='A legend',
        value='A is in yellow',
        style={'width': '100%', 'height': 50,'width':250,'background':'yellow','fontSize':20}),
        dcc.Textarea(
        id='B legend',
        value='B is in blue',
        style={'width': '100%', 'height': 50,'width':250,'background':'blue','fontSize':20}),
        ])
])

@app.callback(
    Output("refresh-graph-interval","max_intervals"),
    Output("refresh-graph-interval","n_intervals"),
    Input('submit-val','n_clicks'),
    
    State('Step','value'), prevent_initial_call=True)
def update_interval(submit, step):
    
    return step,0
    
    


#Rebuild the plot according to the value of A and B

    
   
@app.callback(
    Output(component_id='slider-output-distribution',component_property='children'),
    Output("refresh-graph-interval","disabled"),
    Output('graph','figure'),
    Output('A','value'),
    Output('B','value'),
    Output('storeSquare','data'),
    Input("refresh-graph-interval","n_intervals"),
    Input('submit-val','n_clicks'),
    Input('stop-val','n_clicks'),
    State('A', 'value'),
    State("B","value"),
    State('fitness','value'),
    State('negfit','value'),
    State('storeSquare','data'),prevent_initial_call=True
    )
def update_graph(n, submit, stop, A, B, fitness, selection_on, storeSquare):
    
    ctx = dash.callback_context
    button = ctx.triggered[0]['prop_id'].split('.')[0]
    if button == 'stop-val':
        fig=px.imshow(img,binary_string=True)
        fig.update_layout(coloraxis_showscale=False)
        return " ", True, fig, A, B, storeSquare
    elif button == "submit-val":
        if not isPerfectsquare(B+A):
            fig=px.imshow(img)
            fig.update_layout(coloraxis_showscale=False)
            return "Please change the numbers to form a square",True, fig, A, B, storeSquare
        else:
            newSquare=CreateSquare(A,B)
            fig=px.imshow(storeSquare)
            fig.update_layout(coloraxis_showscale=False)
            return " ", False, fig, A, B, newSquare
    else:
            
        if (A==0):
            newSquare = np.array([[[13, 8, 135], [13, 8, 135]],
                        [[13, 8, 135], [13, 8, 135]]
                    ], dtype=np.uint8)
            fig=px.imshow(newSquare)
            fig.update_layout(coloraxis_showscale=False)
        elif(B==0):
            newSquare = np.array([[[240, 249, 33], [240, 249, 33]],
                        [[240, 249, 33], [240, 249, 33]]
                    ], dtype=np.uint8)
            fig=px.imshow(newSquare)
            fig.update_layout(coloraxis_showscale=False)
            
        else:
            A, B, toBirth, toDie = CalculateNextStep(A,B,fitness,selection_on)
            newSquare=storeSquare
            newSquare=Updatesquare(newSquare, toBirth, toDie, A, B)
            fig=px.imshow(newSquare)
            fig.update_layout(title_text="Step "+str(n),
                title_font_size=30)
            fig.update_layout(coloraxis_showscale=False)
            time.sleep(0.1)
        return " ", False,fig, A, B, newSquare
        



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')  