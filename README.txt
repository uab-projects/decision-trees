#==========================================#
# Instruccions per executar el codi        #
#==========================================#
1. Obrir una terminal en el directori on es troba aquest README.txt i el codi
  (carpetes src, res, cnf)
2. Executar el programa amb Python 3
   python src/main.py -h
3. Es mostrara per pantalla les opcions disponibles per a passar al programa
   Les mes comuns son:
    -d <dataset>
	Especificar el dataset a carregar (cerca dins la carpeta res/dataset)
	Tenim nomes dos, adult i mushroom
    -a <algorisme>
	Algorisme a usar (id3, c4.5 o dummy)
	-c <classificador>
	Especificar el classificador, el numero de columna o be el nom
	En el nostre cas, -c will-die o -c 0 per a mushroom
	i -c salary o -c 14 per a adult
	Per a mes diversio per consola, usar -h i prendre amb moderacio

	Tipicament per a mushroom
	python src/main.py -d mushroom

	I per a adults:
	python src/main.py -d adult -c salary

	L'algorisme per defecte és C4.5 i no cal especificar-lo si no es precisa
	cap canvi.
