# PlateCost


To run our code use these commands in the terminal: 
pip install python-multipart (For file uploads)
pip install -r requirements.txt (Download dependencies in that file)
uvicorn api:app --reload (To run the server)


Our app basically lets small local restaurants upload receipts of their spendings and then we use tinyllama to analyze and categorize these foods, after that. For the receipts to be read, we use OCR to read the image and create a JSON object that will turn all the purchases into text with the item and its price. On the dashboard page, we have a pie chart displaying all the spendings in different categories and when you select a category it will show the spending trends in a graph, we then will have ai insights on spending trends.


Were gonna have test receipt txt files and a receipt image that both can be used to upload and analyze.

Many functions might not have the ai comment over it however we both have line suggestions activated for more of a syntax or fixing the formating of texts aspect then a logic one.
