### 1. Describe how the client could use the model to improve its debt collection strategy

Great part of the problem with debt collection today relies in two main factors:

1. Assign this kind of work to third-party agencies, which leads to inneficient management focus and lack of transparency;
2. Limited availability of tools, such as a credit policy, machine learning models and automated processes.


Since this model can take into account features such as demography, historical debt behavior, price changes, and so on, 
a machine learning would help suggest a credit-score policy, identifying risky and vulnerable clients,
adjusting discount or postpone payment terms and installments accordingly. A paymment plan
can be personalized and product suggestions more effective.

It could also be used to build a customer lifecycle:

1. Acquisition: Enrollment of new customers, historical background to ensure good new clients, product suggestions;
2. Evalluation: daily monitoring of clients, triggering evidences of "pre-late" payments and act preemptively, 
not punitively;
3. Automation: easier payment methods (digitalization), avoiding billing issues. Data aggregation in a whole dedicated environment;
4. Collection: educate customers on good credit scores, client's segmentation by payments, demography.


### 2. Which additional data you think would most add predictive power to the model? Explain why.
### 3. How would you access this data?

Weather conditions: Eletricity demand may have a seasonal component such as season of the year, impacting average
temperature. (A trusted API service, such as Google Weather, OpenWeather or WeatherAPI, or even spacial agencies
if available)

Regulatory taxes: Although more fixed, it still has periodic increases in prices. (Official State's channels of communication)

Inflation: an index that affects everything in all economies. (Official State's channels of communication)

Transmission, distribution and generation costs: Although more constant, it still has periodic changes in prices as well.
(Probably a data the client already has)

Commodities prices: depending on the source or destiny of energy. E.g., the few big clients may increase or decrease
their energy use depending on how commodities prices change. (Probably a data the client already has, but also market
indexes)

Financial Credit Score: As we saw, "total_debt" value and "total_bills" were the two most important features to predict 
debt colletion. So more information about customer's behavior would play a decent role.
