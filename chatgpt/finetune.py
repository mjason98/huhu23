import openai
import pandas as pd
from tqdm import tqdm
import json

openai.api_key = ''

system_text = """Eres un sistema encargado de clasificar los textos como humorísticos o no, respondiendo con Sí o No.
Ten en cuenta que los textos pueden ser hirientes y que son informales, ya que la mayoría son tuits.
"""


def create_file():
    filel = openai.File.list()['data']

    for f in filel:
        print(f)

        sid = f['id']

        dele = openai.File.delete(sid)

        print(dele)

    file = openai.File.create(
        file=open("data_example.jsonl", "rb"),
        purpose='fine-tune',
        user_provided_filename='huhu23_train'
    )

    print('# File created')
    print(file)


def make_dataset(name: str):
    data = pd.read_csv(name)

    with open('data_example.jsonl', 'a') as file:
        iter = tqdm(range(len(data)))
        for i in iter:
            text = data.loc[i, 'tweet']
            label = data.loc[i, 'humor']

            label = "Sí." if label == 1 else "No."

            format_s = {"messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text},
                {"role": "assistant", "content": label}
            ]}

            format_s = json.dumps(format_s) + "\n"

            file.write(format_s)


def fine_tune():
    filel = openai.File.list()

    print(filel)

    file_id = "file-oyoz9BmUbnKlp7iefwack97C"

    openai.FineTuningJob.create(
            training_file=file_id,
            model="gpt-3.5-turbo")

    ft = openai.FineTuningJob.list(limit=1)

    print(ft)


def list_jobs():
    info = openai.FineTuningJob.retrieve("ftjob-APYlLvMKCAfKMf5bIW7gymTv")
    print(info)


if __name__ == "__main__":
    make_dataset('train.csv')
    create_file()
    fine_tune()
    list_jobs()
