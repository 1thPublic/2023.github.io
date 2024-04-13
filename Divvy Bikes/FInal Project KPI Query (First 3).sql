--===================================================
-------------1.Ride duration
--==================================================
USE WAREHOUSE my_wh;
-- Average & max durations of each ride (in minutes) 
SELECT
    ROUND(AVG(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)),2) AS Average_Duration_Minutes,
    ROUND(MAX(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)),2) AS Max_Duration_Minutes
FROM
    TRIP_FACTS a
JOIN TRIP_TIME_DIM b ON a.Ride_ID = b.Ride_ID;

-- Average monthly/yearly/daily ride duration
SELECT
    MONTH(b.Started_At) AS Month,
    ROUND(AVG(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Average_Duration_Minutes,
    ROUND(MAX(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Max_Duration_Minutes
FROM
    TRIP_FACTS a
JOIN TRIP_TIME_DIM b ON a.Ride_ID = b.Ride_ID
GROUP BY MONTH(Started_At)
ORDER BY MONTH(Started_At);

-- Average ride duration for each day of week
SELECT
    DAYOFWEEK(Started_At) AS DayOfWeekIndex,
    CASE
        WHEN DAYOFWEEK(Started_At) = 0 THEN 'Sunday'
        WHEN DAYOFWEEK(Started_At) = 1 THEN 'Monday'
        WHEN DAYOFWEEK(Started_At) = 2 THEN 'Tuesday'
        WHEN DAYOFWEEK(Started_At) = 3 THEN 'Wednesday'
        WHEN DAYOFWEEK(Started_At) = 4 THEN 'Thursday'
        WHEN DAYOFWEEK(Started_At) = 5 THEN 'Friday'
        WHEN DAYOFWEEK(Started_At) = 6 THEN 'Saturday'
    END AS DayOfWeek,
    ROUND(AVG(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Average_Duration_Minutes,
    ROUND(MAX(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Max_Duration_Minutes
FROM
    TRIP_FACTS a
JOIN TRIP_TIME_DIM b ON a.Ride_ID = b.Ride_ID
GROUP BY
    DAYOFWEEK(Started_At)
ORDER BY
    DAYOFWEEK(Started_At);

-- The average duration of each ride per hour of day (use start time)
SELECT
    HOUR(b.Started_At) AS HourOfDay,
    ROUND(AVG(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Average_Duration_Minutes,
    ROUND(MAX(TIMESTAMPDIFF(MINUTE, Started_At, Ended_At)), 2) AS Max_Duration_Minutes
FROM
    TRIP_FACTS a
JOIN TRIP_TIME_DIM b ON a.Ride_ID = b.Ride_ID
GROUP BY
    HOUR(Started_At)
ORDER BY
    HOUR(Started_At);

--===================================================
-------------2. Most common rideable type
--===================================================
-- Most common rideable type overall
SELECT    
    Rideable_Type AS Most_Common_Rideable_Type,
    COUNT(*) AS Counts
FROM
    TRIP_FACTS a
JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
GROUP BY
    Rideable_Type
ORDER BY
    Counts DESC
LIMIT 1;

-- Most common rideable type per month
WITH Monthly_Rideable_Counts AS (
    SELECT 
        MONTH(c.Started_At) AS Month, 
        b.Rideable_Type, 
        COUNT(*) AS Counts,
        RANK() OVER (PARTITION BY MONTH(c.Started_At) ORDER BY COUNT(*) DESC) AS ranking
    FROM TRIP_FACTS a
    JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
    JOIN TRIP_TIME_DIM c ON a.Ride_ID = c.Ride_ID
    GROUP BY MONTH(c.Started_At), b.Rideable_Type
),
Ranked_Monthly_Rideable AS (
    SELECT 
        Month, 
        Rideable_Type, 
        Counts
    FROM Monthly_Rideable_Counts
    WHERE ranking = 1
)
SELECT 
    Month, 
    Rideable_Type AS Most_Common_Rideable_Type, 
    Counts
FROM Ranked_Monthly_Rideable
ORDER BY Month;

-- Most common rideable type per day of week
WITH Daily_Rideable_Counts AS (
    SELECT 
        DAYOFWEEK(c.Started_At) AS Day_Of_Week, 
        b.Rideable_Type, 
        COUNT(*) AS Counts,
        RANK() OVER (PARTITION BY DAYOFWEEK(c.Started_At) ORDER BY COUNT(*) DESC) AS ranking
    FROM 
        TRIP_FACTS a
    JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
    JOIN TRIP_TIME_DIM c ON a.Ride_ID = c.Ride_ID
    GROUP BY DAYOFWEEK(c.Started_At), b.Rideable_Type
),
Ranked_Daily_Rideable AS (
    SELECT 
        Day_Of_Week, 
        Rideable_Type, 
        Counts
    FROM Daily_Rideable_Counts
    WHERE ranking = 1
)
SELECT 
    CASE
        WHEN Day_Of_Week = 0 THEN 'Sunday'
        WHEN Day_Of_Week = 1 THEN 'Monday'
        WHEN Day_Of_Week = 2 THEN 'Tuesday'
        WHEN Day_Of_Week = 3 THEN 'Wednesday'
        WHEN Day_Of_Week = 4 THEN 'Thursday'
        WHEN Day_Of_Week = 5 THEN 'Friday'
        WHEN Day_Of_Week = 6 THEN 'Saturday'
    END AS Day_Of_Week_Name,
    Rideable_Type AS Most_Common_Rideable_Type, 
    Counts
FROM Ranked_Daily_Rideable
ORDER BY Day_Of_Week;

-- Most common rideable type per hour of day
WITH Hourly_Rideable_Counts AS (
    SELECT 
        HOUR(c.Started_At) AS Hour_Of_Day, 
        b.Rideable_Type, 
        COUNT(*) AS Counts,
        RANK() OVER (PARTITION BY HOUR(c.Started_At) ORDER BY COUNT(*) DESC) AS ranking
    FROM TRIP_FACTS a
    JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
    JOIN TRIP_TIME_DIM c ON a.Ride_ID = c.Ride_ID
    GROUP BY HOUR(c.Started_At), b.Rideable_Type
),
Ranked_Hourly_Rideable AS (
    SELECT 
        Hour_Of_Day, 
        Rideable_Type, 
        Counts
    FROM Hourly_Rideable_Counts
    WHERE ranking = 1
)
SELECT 
    Hour_Of_Day,
    Rideable_Type AS Most_Common_Rideable_Type, 
    Counts
FROM Ranked_Hourly_Rideable
ORDER BY Hour_Of_Day;

-- Most common rideable per user type
WITH UserType_Rideable_Counts AS (
    SELECT 
        c.User_Type,
        b.Rideable_Type, 
        COUNT(*) AS Counts,
        RANK() OVER (PARTITION BY c.User_Type ORDER BY COUNT(*) DESC) AS ranking
    FROM 
        TRIP_FACTS a
    JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
    JOIN USER_TYPE_DIM c ON a.USER_TYPE_KEY = c.USER_TYPE_KEY
    GROUP BY c.User_Type, b.Rideable_Type
),
Ranked_UserType_Rideable AS (
    SELECT 
        User_Type, 
        Rideable_Type, 
        Counts
    FROM UserType_Rideable_Counts
    WHERE ranking = 1
)
SELECT 
    User_Type,
    Rideable_Type AS Most_Common_Rideable_Type, 
    Counts
FROM Ranked_UserType_Rideable
ORDER BY User_Type;

-- The most common rideable type for each station
WITH Station_Rideable_Counts AS (
    SELECT 
        Station_ID,
        Station_Name, 
        Rideable_Type, 
        COUNT(*) AS Counts,
        RANK() OVER (PARTITION BY Station_Name ORDER BY COUNT(*) DESC) AS ranking
    FROM 
        TRIP_FACTS a
    JOIN RIDEABLE_TYPE_DIM b ON a.RIDEABLE_TYPE_KEY = b.RIDEABLE_TYPE_KEY
    JOIN LOCATION_DIM c ON a.START_STATION_KEY = c.STATION_KEY
    GROUP BY Station_ID, Station_Name, Rideable_Type
),
Ranked_Station_Rideable AS (
    SELECT 
        Station_ID,
        Station_Name,
        Rideable_Type, 
        Counts
    FROM Station_Rideable_Counts
    WHERE ranking = 1
)
SELECT 
    Station_ID,
    Station_Name,
    Rideable_Type AS Most_Common_Rideable_Type, 
    Counts
FROM 
    Ranked_Station_Rideable
ORDER BY Station_ID;

--===================================================
-------------3. Most frequented route
--===================================================
-- Most frequented route overall
SELECT 
    start_station.STATION_NAME AS Start_Station_Name, 
    end_station.STATION_NAME AS End_Station_Name, 
    COUNT(*) AS Trip_Count
FROM TRIP_FACTS
JOIN LOCATION_DIM AS start_station ON TRIP_FACTS.start_station_key = start_station.STATION_KEY
JOIN LOCATION_DIM AS end_station ON TRIP_FACTS.end_station_key = end_station.STATION_KEY
GROUP BY start_station.STATION_NAME, end_station.STATION_NAME
ORDER BY Trip_Count DESC
LIMIT 3;

-- Most frequented route by month
WITH MonthlyRouteCounts AS (
    SELECT 
        MONTH(Started_At) AS Month,
        start_station.STATION_NAME AS Start_Station_Name, 
        end_station.STATION_NAME AS End_Station_Name, 
        COUNT(*) AS Trip_Count,
        RANK() OVER (PARTITION BY MONTH(Started_At) ORDER BY COUNT(*) DESC) AS Rank
    FROM TRIP_FACTS
    JOIN LOCATION_DIM AS start_station ON TRIP_FACTS.start_station_key = start_station.STATION_KEY
    JOIN LOCATION_DIM AS end_station ON TRIP_FACTS.end_station_key = end_station.STATION_KEY
    JOIN TRIP_TIME_DIM ON TRIP_FACTS.Ride_ID = TRIP_TIME_DIM.Ride_ID
    GROUP BY Month, Start_Station_Name, End_Station_Name
)
SELECT  
    Month, 
    Start_Station_Name, 
    End_Station_Name, 
    Trip_Count
FROM MonthlyRouteCounts
WHERE Rank = 1
ORDER BY Month;

-- Most frequented route by day of week
WITH Daily_Route_Counts AS (
    SELECT 
        DAYOFWEEK(TRIP_TIME_DIM.Started_At) AS Day_Of_Week,
        start_station.STATION_NAME AS Start_Station_Name, 
        end_station.STATION_NAME AS End_Station_Name, 
        COUNT(*) AS Trip_Count,
        RANK() OVER (PARTITION BY DAYOFWEEK(TRIP_TIME_DIM.Started_At) ORDER BY COUNT(*) DESC) AS Rank
    FROM TRIP_FACTS
    JOIN LOCATION_DIM AS start_station ON TRIP_FACTS.start_station_key = start_station.STATION_KEY
    JOIN LOCATION_DIM AS end_station ON TRIP_FACTS.end_station_key = end_station.STATION_KEY
    JOIN TRIP_TIME_DIM ON TRIP_FACTS.Ride_ID = TRIP_TIME_DIM.Ride_ID
    GROUP BY Day_Of_Week, Start_Station_Name, End_Station_Name
)
SELECT  
    CASE
        WHEN Day_Of_Week = 0 THEN 'Sunday'
        WHEN Day_Of_Week = 1 THEN 'Monday'
        WHEN Day_Of_Week = 2 THEN 'Tuesday'
        WHEN Day_Of_Week = 3 THEN 'Wednesday'
        WHEN Day_Of_Week = 4 THEN 'Thursday'
        WHEN Day_Of_Week = 5 THEN 'Friday'
        WHEN Day_Of_Week = 6 THEN 'Saturday'
    END AS Day_Of_Week_Name,
    Start_Station_Name, 
    End_Station_Name, 
    Trip_Count
FROM Daily_Route_Counts
WHERE Rank = 1
ORDER BY Day_Of_Week;


-- Most frequented route by hour of day
WITH Hourly_Route_Counts AS (
    SELECT 
        HOUR(TRIP_TIME_DIM.Started_At) AS Hour_Of_Day,
        start_station.STATION_NAME AS Start_Station_Name, 
        end_station.STATION_NAME AS End_Station_Name, 
        COUNT(*) AS Trip_Count,
        RANK() OVER (PARTITION BY HOUR(TRIP_TIME_DIM.Started_At) ORDER BY COUNT(*) DESC) AS Rank
    FROM TRIP_FACTS
    JOIN LOCATION_DIM AS start_station ON TRIP_FACTS.start_station_key = start_station.STATION_KEY
    JOIN LOCATION_DIM AS end_station ON TRIP_FACTS.end_station_key = end_station.STATION_KEY
    JOIN TRIP_TIME_DIM ON TRIP_FACTS.Ride_ID = TRIP_TIME_DIM.Ride_ID
    GROUP BY Hour_Of_Day, Start_Station_Name, End_Station_Name
)
SELECT  
    Hour_Of_Day,
    Start_Station_Name, 
    End_Station_Name, 
    Trip_Count
FROM Hourly_Route_Counts
WHERE Rank = 1
ORDER BY Hour_Of_Day;

-- Most frequented route per user type
WITH User_Type_Route_Counts AS (
    SELECT 
        user_type_dim.User_Type,
        start_station.STATION_NAME AS Start_Station_Name, 
        end_station.STATION_NAME AS End_Station_Name, 
        COUNT(*) AS Trip_Count,
        RANK() OVER (PARTITION BY user_type_dim.User_Type ORDER BY COUNT(*) DESC) AS Rank
    FROM TRIP_FACTS a
    JOIN LOCATION_DIM AS start_station ON a.start_station_key = start_station.STATION_KEY
    JOIN LOCATION_DIM AS end_station ON a.end_station_key = end_station.STATION_KEY
    JOIN USER_TYPE_DIM AS user_type_dim ON a.USER_TYPE_KEY = user_type_dim.USER_TYPE_KEY
    GROUP BY user_type_dim.User_Type, Start_Station_Name, End_Station_Name
)
SELECT  
    User_Type,
    Start_Station_Name, 
    End_Station_Name, 
    Trip_Count
FROM User_Type_Route_Counts
WHERE Rank = 1
ORDER BY User_Type;

-- Most frequented route per rideable type
WITH RideableTypeRouteCounts AS (
    SELECT 
        rideable_type_dim.Rideable_Type,
        start_station.STATION_NAME AS Start_Station_Name, 
        end_station.STATION_NAME AS End_Station_Name, 
        COUNT(*) AS Trip_Count,
        RANK() OVER (PARTITION BY rideable_type_dim.Rideable_Type ORDER BY COUNT(*) DESC) AS Rank
    FROM TRIP_FACTS a
    JOIN LOCATION_DIM AS start_station ON a.start_station_key = start_station.STATION_KEY
    JOIN LOCATION_DIM AS end_station ON a.end_station_key = end_station.STATION_KEY
    JOIN RIDEABLE_TYPE_DIM AS rideable_type_dim ON a.RIDEABLE_TYPE_KEY = rideable_type_dim.RIDEABLE_TYPE_KEY
    GROUP BY rideable_type_dim.Rideable_Type, Start_Station_Name, End_Station_Name
)
SELECT  
    Rideable_Type,
    Start_Station_Name, 
    End_Station_Name, 
    Trip_Count
FROM RideableTypeRouteCounts
WHERE Rank = 1
ORDER BY Rideable_Type;




