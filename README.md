# LSTM with DL4J in Scala


This projects aim is to evaluate the DL4J Platform with Scala.

**Data**

The data is available [here](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line).
It shows some intersting periodic characteristics over time, while a clear trend is visible:

![Screenshot](international-airline-passengers.png "Training Data")

**Results**

My LSTM fits the training data quite well (wow!...), but isn't quite good in predicting the test data:

![Screenshot](epoch0.png "Initial Prediction")

![Screenshot](epoch100.png "Initial Prediction")

![Screenshot](epoch500.png "Initial Prediction")

Also you can see the metrics from DL4J UI-Server:
![Screenshot](ui_server.png "Initial Prediction")
