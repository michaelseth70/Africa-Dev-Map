import dash
from dash import html, dcc, Dash, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import openai
import logging
import traceback
from openai.error import RateLimitError, OpenAIError
import os  # Import os to handle environment variables
import time  # Import time for rate limit handling


# Configure logging to show errors in CMD
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


# Set your OpenAI API key securely using environment variables
# Ensure the environment variable 'OPENAI_API_KEY' is set in your system
openai.api_key = os.getenv('OPENAI_API_KEY')


# List of 55 African countries
african_countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
    'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
    'Democratic Republic of the Congo', 'Republic of the Congo', "Cote d'Ivoire",
    'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia',
    'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho',
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius',
    'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe',
    'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan',
    'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]


# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("African Development Initiatives Explorer", className='text-center mb-4'), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Input(id='topic-input', placeholder='Enter a development topic...', type='text'),
            dbc.Button('Search', id='search-button', color='primary', className='mt-2')
        ], width=6),
    ], justify='center'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Initiatives Data Table", className='mt-4'),
            dbc.Table(id='data-table', bordered=True, hover=True, responsive=True, striped=True)
        ], width=12)
    ])
], fluid=True)


# Callback to update map and table based on input topic
@app.callback(
    [Output('heatmap', 'figure'),
     Output('data-table', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('topic-input', 'value')]
)
def update_output(n_clicks, topic):
    if n_clicks is None or not topic:
        return dash.no_update, dash.no_update


    try:
        processed_topic = process_topic_with_openai(topic)
        initiatives_data = fetch_initiatives_data(processed_topic)
        fig = create_heatmap(initiatives_data)
        table = create_data_table(initiatives_data, processed_topic)
        return fig, table
    except Exception as e:
        logging.error("An error occurred:", exc_info=True)
        error_row = html.Tr([html.Td("Error fetching data. Check logs for details.", colSpan=2)])
        table_header = html.Thead(html.Tr([html.Th("Country"), html.Th("Initiatives Summary")]))
        return dash.no_update, [table_header, html.Tbody([error_row])]


def process_topic_with_openai(topic):
    """
    Process the topic using OpenAI Chat API to make sense of it in development terms.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a development expert. Provide clear insights on the topic."},
                {"role": "user", "content": f"Explain the development topic: {topic}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error processing topic with OpenAI: {e}")
        raise


def fetch_initiatives_data(processed_topic):
    """
    Simulate fetching initiatives data for each country based on the processed topic.
    Replace this function with actual data fetching logic.
    """
    data = []
    for country in african_countries:
        num_initiatives = min(50, abs(hash(country + processed_topic)) % 51)
        initiatives = [f"{processed_topic} Initiative {i+1} in {country}" for i in range(num_initiatives)]
        data.append({
            'country': country,
            'num_initiatives': num_initiatives,
            'initiatives': initiatives
        })
    return data


def create_heatmap(data):
    """
    Create an interactive heatmap showing the number of initiatives per country.
    """
    df = pd.DataFrame(data)
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='num_initiatives',
        hover_name='country',
        color_continuous_scale='Blues',
        title='Number of Initiatives per Country'
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig


def create_data_table(data, topic):
    """
    Create a data table displaying a narrative synthesis of initiatives per country.
    """
    table_header = html.Thead(html.Tr([html.Th("Country"), html.Th("Initiatives Summary")]))
    rows = []
    for entry in data:
        try:
            synthesis = generate_country_synthesis(entry['country'], entry['initiatives'], topic)
            rows.append(html.Tr([html.Td(entry['country']), html.Td(synthesis)]))
        except Exception as e:
            logging.error(f"Error generating synthesis for {entry['country']}: {e}")
            rows.append(html.Tr([html.Td(entry['country']), html.Td("Error generating summary.")]))
            continue
    table_body = html.Tbody(rows)
    return [table_header, table_body]


def generate_country_synthesis(country, initiatives, topic):
    """
    Generate a narrative synthesis of initiatives for a given country.
    """
    try:
        # Prepare a prompt summarizing the initiatives for the country
        initiatives_text = "; ".join(initiatives)
        prompt = (
            f"Provide a concise narrative synthesis (maximum 150 words) of the following development initiatives in {country} related to '{topic}':\n\n"
            f"{initiatives_text}\n\n"
            "Focus on key themes, progress, and impact."
        )


        # OpenAI API call with retry logic for rate limits
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a skilled analyst specializing in development initiatives. Generate concise summaries based on provided data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,  # Adjust to stay within 150 words
                    temperature=0.5
                )
                synthesis = response.choices[0].message.content.strip()
                return synthesis
            except RateLimitError:
                logging.warning(f"Rate limit exceeded for {country}. Retrying...")
                time.sleep(2)  # Wait before retrying
            except Exception as e:
                logging.error(f"Error during OpenAI API call for {country}: {e}")
                raise
        return "Unable to generate summary due to API rate limits."
    except Exception as e:
        logging.error(f"Error generating synthesis for {country}: {e}")
        return "Error generating summary."


if __name__ == '__main__':
    app.run_server(debug=True)
