
from h2o_wave import main, app, Q, ui, on, run_on, data
from typing import Optional, List
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score




def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card



def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return

    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)

#..................................................................................................................
@on('#page1')
async def page1(q: Q):
    q.page['sidebar'].value = '#page1'
    clear_cards(q)  
    add_card(q, 'article', ui.tall_article_preview_card(
        box=ui.box('vertical', height='600px'), title='Student Stress Detection System',
        image='https://www.sbs.com.au/news/sites/sbs.com.au.news/files/podcast_images/stressed_students_in_the_library_getty.jpg',
        content='''
Welcome to our Student Stress Detection System! We understand the importance of mental well-being in academic success. my innovative system utilizes advanced Machine Learing techniques to detect and monitor stress levels among students. By gathering various indicators and patterns, we provide valuable insights to educators and counselors, enabling them to offer timely support and interventions. Together, let's create a nurturing environment where students can thrive both academically and emotionally,if the student fill the form the system show the stress level ......
        '''
))

#.......................................................................................................................
@on('#page2')
async def page2(q: Q):
    df = pd.read_csv('StressLevelDataset.csv')
    q.page['sidebar'].value = '#page2'
    clear_cards(q) 
    add_card(q,'form',ui.form_card(box = 'vertical', items=[
        ui.text(content='<div style="color: blue; font-size: 25px; font-weight: bold;">DataSet for Detect Stress </div>',),
        ui.text(f'Total rows: {len(df)}'),
            ui.text(f'Total columns: {len(df.columns)}'),
            ui.table(
                name='my_table',
                columns=[ui.table_column(name=str(col), label=str(col)) for col in df.columns],
                rows=[ui.table_row(name=str(i), cells=[str(df[col][i]) for col in df.columns]) for i in range(len(df))],
                height='500px'
            )
    ])) 


#........................................................................................................

@on('#page3')

async def page1(q: Q):
   
    data = pd.read_csv('StressLevelDataset.csv')  
    q.page['sidebar'].value = '#page3'
    clear_cards(q) 
    if not q.client.processed:

        X = data.drop('stress_level', axis=1)  
        y = data['stress_level']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        add_card(q,'table',ui.form_card(box = 'vertical', items=[ 
            ui.text('Use RandomForest Classifiers for Train the Data'),
            ui.text(f'Test Accuracy: {accuracy:.2f}'),
            ]))
#.......................................................................................................................
@on('#page4')
async def page3(q: Q):
    q.page['sidebar'].value = '#page4'
    clear_cards(q)  #to remove the card details
    add_card(q,'form',ui.form_card(box = 'vertical', items=[
        ui.text(content='<div style="color: blue; font-size: 25px; font-weight: bold;">Students!  Fill The Details </div>',),
        ui.textbox(name = 'textbox1',label='Name'),
        ui.textbox(name = 'textbox2',label='Anxiety_level 0-25 '),
        ui.textbox(name = 'textbox3',label='Self Esteem 0-30'),
        ui.textbox(name = 'textbox4',label='Mental_health_history 0/1'),
        ui.textbox(name = 'textbox5',label='Depression level 0-30'),
        ui.textbox(name = 'textbox6',label='No of headache In Week'),
        ui.textbox(name = 'textbox7',label='blood_pressure rate 0/1/2/3'),
        ui.textbox(name = 'textbox8',label='Sleep Quality rate 1-5'),
        ui.textbox(name = 'textbox9',label='Breathing_problem rate 1-5'),
        ui.textbox(name = 'textbox10',label='Noise Level rate 1-5'),
        ui.textbox(name = 'textbox11',label='Living Conditions rate 1-5'),
        ui.textbox(name = 'textbox12',label='Safety rate 1-5'),
        ui.textbox(name = 'textbox13',label='Basic Needs rate 1-5'),
        ui.textbox(name = 'textbox14',label='Academic_performance rate 1-5'),
        ui.textbox(name = 'textbox15',label='Study_load rate 1-5'),
        ui.textbox(name = 'textbox16',label='Teacher Student RelationShip rate 1-5'),
        ui.textbox(name = 'textbox17',label='Future Career Concern rate 1-5'),
        ui.textbox(name = 'textbox18',label='Social Support rate 1-5'),
        ui.textbox(name = 'textbox19',label='Peer_Pressure rate 1-5'),
        ui.textbox(name = 'textbox20',label='Extra Curricular Activities rate 1-5'),
        ui.textbox(name = 'textbox21',label='Bullying rate 1-5'),
        ui.button(name='button1', label ='Submit'),
    ]))
    @on()
    async def button1(q: Q):
        # Handle form submission
        clear_cards(q) 
        data = pd.read_csv('StressLevelDataset.csv') 
       

        X = data.drop('stress_level', axis=1)  
        y = data['stress_level']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train) 

        user_inputs = {
            'anxiety_level': [q.args['textbox2']],  # Ensure input is converted to int
            'self_esteem': [q.args['textbox3']],
            'mental_health_history': [q.args['textbox4']],
            'depression': [q.args['textbox5']],
            'headache': [q.args['textbox6']],
            'blood_pressure': [q.args['textbox7']],
            'sleep_quality': [q.args['textbox8']],
            'breathing_problem': [q.args['textbox9']],
            'noise_level': [q.args['textbox10']],
            'living_conditions': [q.args['textbox11']],
            'safety': [q.args['textbox12']],
            'basic_needs': [q.args['textbox13']],
            'academic_performance': [q.args['textbox14']],
            'study_load': [q.args['textbox15']],
            'teacher_student_relationship': [q.args['textbox16']],
            'future_career_concerns': [q.args['textbox17']],
            'social_support': [q.args['textbox18']],
            'peer_pressure': [q.args['textbox19']],
            'extracurricular_activities': [q.args['textbox20']],
            'bullying': [q.args['textbox21']],
           
          
        }
        user_input_df = pd.DataFrame(user_inputs)
       
        predicted_stress_level = rf_classifier.predict(user_input_df)[0]

        
       
        add_card(q,'table',ui.form_card(box = 'vertical', items=[ 
            ui.text('Hai ! '),
            ui.text(q.args.textbox1),
            ui.text('Your Stress Level is :'),
            ui.text(f'{predicted_stress_level}'),
            ]))


#..........................................................................................
async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
        ui.zone('main', size='1', direction=ui.ZoneDirection.ROW, zones=[
            ui.zone('sidebar', size='250px'),
            ui.zone('body', zones=[
                ui.zone('header'),
                ui.zone('content', zones=[
                    ui.zone('horizontal', direction=ui.ZoneDirection.ROW),
                    ui.zone('vertical'),
                    ui.zone('grid', direction=ui.ZoneDirection.ROW, wrap='stretch', justify='center')
                ]),
            ]),
        ])
    ])])
    q.page['sidebar'] = ui.nav_card(
        box='sidebar', color='primary', title='', subtitle="",
        value=f'#{q.args["#"]}' if q.args['#'] else '#page1'
       , items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#page1', label='Home'),
                ui.nav_item(name='#page2', label='Show Data'),
                ui.nav_item(name='#page3', label='Processing'),
                ui.nav_item(name='#page4', label='Find The Stress Level'),
            
            ]),
        ])
    q.page['header'] = ui.header_card(
        box='header', title='Student Stress Detection', subtitle='',
        
        items=[
            ui.persona(title='Luxshika P', subtitle='Developer', size='xs',
                       image='https://cdn.dribbble.com/users/2131993/screenshots/15628402/media/7bb0d27e44d8c2eff47276ae86bfd6a3.png?compress=1&resize=400x300'),
        ]
    )
    
    if q.args['#'] is None:
        await page1(q)


@app('/dashboard')
async def serve(q: Q):
    
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

   
    await run_on(q)
    await q.page.save()

