import logging
from datetime import date

import click

from duration_prediction.train import train

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@click.command()
@click.option('--train-month', required=True, help='Training month in YYYY-MM format')
@click.option('--validation-month', required=True, help='Validation month in YYYY-MM format')
@click.option('--model-output-path', required=True, help='Path where the trained model will be saved')
def run(train_month, validation_month, model_output_path):
    # train_month = date(year=2023, month=1, day=1)
    # val_month = date(year=2023, month=2, day=1)
    # model_output_path = './models/2023-01-model.bin'
    train_year, train_month = train_month.split('-')
    train_year = int(train_year)
    train_month = int(train_month)

    val_year, val_month = validation_month.split('-')
    val_year = int(val_year)
    val_month = int(val_month)

    train_month = date(year=train_year, month=train_month, day=1)
    validation_month = date(year=val_year, month=val_month, day=1)

    train(
        train_month=train_month,
        val_month=validation_month,
        model_output_path=model_output_path
    )


if __name__ == '__main__':
    run()
