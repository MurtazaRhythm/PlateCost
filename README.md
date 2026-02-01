# PlateCost


To run our code use these commands in the terminal: 
pip install python-multipart (For file uploads)
pip install -r requirements.txt (Download dependencies in that file)
uvicorn api:app --reload (To run the server)


Our app basically lets small local restaurants upload receipts of their spendings and then we use tinyllama to analyze and categorize these foods, after that. For the receipts to be read, we use OCR to read the image and create a JSON object that will turn all the purchases into text with the item and its price. On the dashboard page, we have a pie chart displaying all the spendings in different categories and when you select a category it will show the spending trends in a graph, we then will have ai insights on spending trends in the insights page.

To see the trends on the graph you need to upload multiple receipts (within the receipt folder)

Were gonna have test receipt txt files and a receipt image that both can be used to upload and analyze.

Some functions might not have the ai comment over it because it might not be ai generated however we both have line suggestions activated for more of a syntax or text parsing aspect then a logic one and those were used.

A big part of the aesthetics of the frontend were prompted because its purely visual and doesn't have much importance with our project concept.
