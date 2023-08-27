import openai
import pandas as pd
from tqdm import tqdm


openai.api_key = ''

system_text = """Eres un sistema encargado de clasificar los textos como humorísticos o no, respondiendo con Sí o No.
Ten en cuenta que los textos pueden ser hirientes y que son informales, ya que la mayoría son tuits.
"""

pairs = (
    """— Hija estas tan gorda que podrías rodar…
    — ¡Abuelo, que grosero! mejor me voy…
    (._.) (|:) (.-.) (:|) (._.) (|:) (.-.) (|:)""",
    "estas tan gordo que cuando pasas por delante de la tele me pierdo los anuncios de antena3"
)


def use_chpt(msg):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {"role": "system", "content": system_text},
             {"role": "user", "content": pairs[0]},
             {"role": "assistant", "content": "Sí."},
             {"role": "user", "content": pairs[1]},
             {"role": "assistant", "content": "Sí."},
             {"role": "user", "content": msg},
            ]
        )

    return response.choices[0].message.content


data_path = "test_gold.csv"


data = pd.read_csv(data_path)

iter = tqdm(range(len(data)))

solution = []

for i in iter:
    ide = data.loc[i, 'index']
    text = data.loc[i, 'tweet']

    sol = "Error"

    for k in range(3):
        try:
            sol = use_chpt(text)
            break
        except Exception as e:
            print("An error occurred:", e)

    solution.append((ide, sol))

df = pd.DataFrame(solution, columns=['index', 'prediction'])
df.to_csv('prediction_gpt_pi_test.csv', index=False)
