import requests
r = requests.get("https://api.github.com/users/Connor-SM")
#print( r )
#print( type( r ))

#Accessing data
#data = r.content
#print(data)

#Converting json to python-understandable view
#data = r.json()
#for k, v in data.items():
#	print(f"Key:{k}\tValue:{v}")
#print(data['name'])

#Passing parameters
r = requests.get("https://api.github.com/search/repositories?q=language:python")
data = r.json()
print(data['total_count'])

