import pandas as pd
import random

random.seed(3)#generate same random numbers every time, number used doesn't matter
names = ["Dinara", "Anelya", "Ayana", "Batyr", "Daniyal"]
ages = [random.randint(0,21) for x in range(len(names))]
people = {"names": names, "ages": ages}
df = pd.DataFrame.from_dict(people)
#print(df)

#Accessing data in DataFrame is divided into 2 types: by column or by record
#1.Accessing data by column
#print(df["ages"])
#print(df["ages"][0])

#2.Accessing data by record: use .loc
#print(df.loc[0])
#print(df.loc[0]["names"])

#Slicing DataFrame
print(df[2:5])
