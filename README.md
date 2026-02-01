# PlateCost


To run our code use these commands in the terminal: 
pip install python-multipart (For file uploads)
pip install -r requirements.txt (Download dependencies in that file)
uvicorn api:app --reload (To run the server)


Our app basically lets small local restaurants upload receipts of their spendings and then we use tinyllama to analyze and categorize these foods, after that. For the receipts to be read, we use OCR to read the image and create a JSON object that will turn all the purchases into text with the item and its price. On the dashboard page, we have a pie chart displaying all the spendings in different categories and when you select a category it will show the spending trends in a graph, we then will have ai insights on spending trends.





PlateCost 🍽️

Simple food spend insights for small restaurants

What is PlateCost?

PlateCost is a lightweight web application that helps small, independent restaurant owners understand where their food money is going by analyzing supplier receipts.

Instead of spreadsheets, accounting software, or POS integrations, PlateCost focuses on one thing only:
turning supplier receipts into clear weekly spending insights.

The Problem

Many small restaurants:

Receive paper or PDF receipts from suppliers

Track expenses manually (or not at all)

Don’t have time for detailed bookkeeping

Just want to know why food costs went up

Existing tools are often:

Too complex

Too expensive

Over-scoped for their real needs

The Solution

PlateCost allows users to:

Upload printed supplier receipts

Automatically extract item-level data using OCR

Categorize items into clear food categories using AI

View weekly spend breakdowns and plain-language insights

No inventory tracking.
No accounting compliance.
No POS setup.

Core Features

📸 Receipt upload (printed, English supplier receipts only)

🔍 OCR extraction of:

Item name

Item price

Vendor name

Purchase date

🧠 AI-powered categorization into fixed categories:

Produce

Meat

Dairy

Dry Goods

Beverages

Other

📊 Weekly spend analysis (rule-based, not AI):

Spend per category

Week-over-week percentage change

Spend concentration by supplier

✍️ AI-generated insights:

One short, clear insight per category

AI explains results, math decides importance

📈 Simple dashboard:

Pie chart of weekly spend by category

Click a category to see spend, change, and insight
