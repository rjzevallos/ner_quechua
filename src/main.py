import util
import argparse
import sys
from tqdm import tqdm
import os
import pandas as pd


def main():
    
    parser = argparse.ArgumentParser("NER quechua")
    parser.add_argument("-t", "--train", help="Dataset de entrenamiento en formato csv", 
        default=None)
    parser.add_argument("-e", "--evaluation", help="Dataset de evaluación en formato csv", default=None)
    parser.add_argument("-u", "--usage", help="Utilizando un archivo csv", default=None)
    parser.add_argument("-i", "--interactive", help="utilizar en modo interactivo", action='store_true')
    parser.add_argument("-o", "--output", help="Archivo csv procesado", default=None)
    args = parser.parse_args()

    root_train = args.train
    root_eval = args.evaluation
    root_usage = args.usage
    root_out = args.output
    
    if args.train and args.evaluation:
        print("Proceso de entrenamiento y evaluación")
        util.train(root_train, root_eval)
        print("Proceso terminado")
    elif args.interactive:
        print("Introduzca la oración a procesar (type 0 to exit):")
        while True:
            in_sent = input()
            if in_sent == '0':
                sys.exit()
            print(util.use_huggingface(in_sent))
    elif args.usage and args.output:
        print("Leyendo el archivo csv")
        data = pd.read_csv(root_usage, sep="\t")
        data_pro = util.use_csv(data)
        print("Guardando csv procesado")
        data_pro.to_csv(args.output+"data_pro.csv", index = False, sep ="\t")
        print("Proceso terminado")


if __name__ == '__main__':
    main()
