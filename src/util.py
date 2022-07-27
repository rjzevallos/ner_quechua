
import pandas as pd
from simpletransformers.ner import NERModel,NERArgs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


mlb = MultiLabelBinarizer()

# Método para conseguir el accuracy y la matris de confusión del modelo NER a entrenar
def get_accuracy_and_confusion (labels, preds):
    pred_list = []
    label_list = []

    for pred in preds:
        pred_list.append(pred[0])
    for label in labels:
        label_list.append(label[0])

    labels = label_list
    preds = pred_list
    label_set = list(set(labels))
    label_set.sort()

    print(label_set)
    matrix = confusion_matrix(labels, preds, labels=label_set)
    print(matrix)
    labels = mlb.fit_transform(labels)
    preds = mlb.transform(preds)

    return accuracy_score(labels, preds)
    
    
# Método que entrena una modelo NER
def train(root_train,root_test):
    train_data = pd.read_csv(root_train)
    test_data = pd.read_csv(root_test)
    label = train_data["labels"].unique().tolist()
    
    args = NERArgs()
    args.num_train_epochs = 10
    args.learning_rate = 1e-4
    args.overwrite_output_dir =True
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.manual_seed = 42
    args.evaluate_during_training = False
    args.save_model_every_epoch = False
    args.save_steps = -1
    model = NERModel('roberta', "Llamacha/QuBERTa" ,labels=label, args =args,use_cuda=True)
    model.train_model(train_data,eval_data = test_data, acc=get_accuracy_and_confusion)
    result, model_outputs, preds_list = model.eval_model(test_data, acc=get_accuracy_and_confusion, output_dir = "/resources")
    print(result)
    print("Guardando model NER")
    print("Modelo NER guardado")
    
    
# Método para usar el modelo NER para un texto    
def use_ner(text):
    model = NERModel("roberta", "Llamacha/ner_quechua")
    prediction, model_output = model.predict([text])
    return prediction
    
    
def use_huggingface(text):
    tokenizer = AutoTokenizer.from_pretrained("Llamacha/ner_quechua")
    model = AutoModelForTokenClassification.from_pretrained("Llamacha/ner_quechua")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    return ner_results

# Método para usar el modelo NER para un archivo CSV
def use_csv(data):
    data["ner"] = data["text"].apply(use_huggingface)
    return data
