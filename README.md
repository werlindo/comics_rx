# comic_rx

**Comic Book ‘Associations’ System**

Build a recommender system to 'prescribe' comics to read/buy/subscribe.

---

**Business Understanding**

I subscribe to comic books at a local comic shop and one thing they do as an intended value-add is something they call ‘associations’.

So for example: “Savage Avengers” is being recommended to those who subscribe to “Savage Sword of Conan”.

After discussing it with the shop manager, it turns out this process is completely human-driven. I’d like to see if I can build a comic recommendation system!

**Data Understanding**
With the help of the shop manager we were able to pull all comic book transactions since they implemented their current point-of-sale system in Feb 2010. It totals about 500K transaction lines. The caveat though is that I have not had the chance to explore the data since it was delivered to me. 

The plan is to build an implicit-ratings matrix based on this information. Because buyers at brick-and-mortar stores don’t usually review their books, we will proxy their ‘review’ with the fact that they purchased the item. The transaction data provides customer id, date of purchase, and specific issue (e.g. Batman #123). There are some limitations that I will discuss further below.

The shop has also offered to further research some potentially useful data, such as metadata on the comics (e.g. Publisher / Genre / Writer / Artist ). Metadata such as this would potentially be useful for supporting the cold-start model.

To augment, I plan to use Amazon review data, from which we can also build a utility matrix. If it turns out the store data is insufficient or faulty then this will become the primary. In the event the Amazon data is insufficient, I l believe there are APIs for comic book metadata communities, such as https://comicbookroundup.com/ or www.comicbookdb.com, where users offer up their own ratings (usually on a scale of 1 to 10).

Data Preparation
I need to perform EDA on the dataset already provided to me and quickly pivot to alternative sources if it is insufficient.
Storage
I will likely use some flavor of AWS RDS to store the structured data. Cleaning will occur via Python and SQL.
If it turns out I will be storing any unstructured data I will likely first turn to use MongoDB Atlas.

Prep
I see two primary tasks:
Getting purchase/review data into the proper form to create an ALS matrix factorization model to be the basis of the recommendation system.
Second, organizing purchase or review data, along with any metadata I can find, to support building a cold-start model.

Modeling
We will holdout out 20% of the purchase/review data for validation; this may be influenced by the breadth of data I am able to procure.
I’ll be using an ALS matrix factorization method to build the base algorithm. I will likely use PySpark for building it.
For cold-start model I think my approach will be influenced by what data will ultimately be available. If I only have purchase data (yes/no), it maybe classification-based? Otherwise, if I end up using review scores, or similar, a regression approach may be more appropriate?
To facilitate testing different approaches I will build a pipeline to iterate through different models.

Evaluation
I plan to use k-fold cross-validation on any parameter tuning, both for ALS model and for the cold-start model. 

With regards to loss metrics, I think if the eventual target variable is a yes/no on to subscribe/buy, then I think an accuracy-related metric would be apt. Otherwise, if the underlying algorithm is dependent on predicting a ‘review score’ I think a metric such as RMSE would be appropriate.

Deployment
My vision for the initial deployment is to have a website for current, or potential, store customer to select their current “pull list” (what comics they subscribe to) and get returned top N recommended titles. 

Eventually, I think it would be great to somehow integrate it into the store’s data/system so store employees can use it to augment their own expert opinion in creating their ‘associations’.

User Story:



