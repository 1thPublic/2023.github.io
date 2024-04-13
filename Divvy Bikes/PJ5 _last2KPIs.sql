--===================================================
-------------Max bike usage (24 Rows)
--==================================================

WITH ride_intervals AS (
    SELECT
        RIDE_ID,
        START_TIMEOFDAY_KEY,
        END_TIMEOFDAY_KEY
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS
),
hourly_rides AS (
    SELECT
        r.RIDE_ID,
        t.HOUR_24
    FROM
        ride_intervals r
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM t
        ON t.TIMEOFDAY_KEY BETWEEN r.START_TIMEOFDAY_KEY AND r.END_TIMEOFDAY_KEY
)
SELECT
    HOUR_24,
    COUNT(DISTINCT RIDE_ID) AS RIDE_COUNT
FROM
    hourly_rides
GROUP BY
    HOUR_24
ORDER BY
    HOUR_24;



--===================================================
-------------Max bike usage TP (3 Rows)
--==================================================

WITH ride_intervals AS (
    SELECT
        RIDE_ID,
        START_TIMEOFDAY_KEY,
        END_TIMEOFDAY_KEY
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS
),
hourly_rides AS (
    SELECT
        r.RIDE_ID,
        t.HOUR_24
    FROM
        ride_intervals r
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM t
        ON t.TIMEOFDAY_KEY BETWEEN r.START_TIMEOFDAY_KEY AND r.END_TIMEOFDAY_KEY
),
hourly_sum AS (
    SELECT CASE
            WHEN HOUR_24 > 4 AND HOUR_24 < 9 THEN 'MorningRushHour'
            WHEN HOUR_24 >= 9 AND HOUR_24 < 16 THEN 'Noon'
            WHEN HOUR_24 >= 16 AND HOUR_24 < 20 THEN 'EveningRushHour'
            ELSE 'Night'
        END AS Time_Period,
        HOUR_24,
        COUNT(DISTINCT RIDE_ID) AS RIDE_COUNT
    FROM
        hourly_rides
    GROUP BY
        Time_Period, HOUR_24
)
SELECT Time_Period,
    ROUND(AVG(RIDE_COUNT), 2) AS AVG_RIDE_COUNT
FROM hourly_sum
GROUP BY Time_Period
ORDER BY
    CASE Time_Period
        WHEN 'MorningRushHour' THEN 1
        WHEN 'Noon' THEN 2
        WHEN 'EveningRushHour' THEN 3
        ELSE 4
    END;



--===================================================
-------------Max bike usage UT (2 Rows)
--==================================================

WITH ride_intervals AS (
    SELECT
        RIDE_ID,
        START_TIMEOFDAY_KEY,
        END_TIMEOFDAY_KEY,
        USER_TYPE_KEY
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS
),
hourly_rides AS (
    SELECT
        r.RIDE_ID,
        u.USER_TYPE,
        t.HOUR_24,
    FROM
        ride_intervals r
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM u
        ON r.USER_TYPE_KEY = u.USER_TYPE_KEY
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM t
        ON t.TIMEOFDAY_KEY BETWEEN r.START_TIMEOFDAY_KEY AND r.END_TIMEOFDAY_KEY
),
hourly_sum AS (
    SELECT USER_TYPE,
        HOUR_24,
        COUNT(DISTINCT RIDE_ID) AS RIDE_COUNT
    FROM
        hourly_rides
    GROUP BY
        USER_TYPE, HOUR_24
)
SELECT USER_TYPE,
    ROUND(AVG(RIDE_COUNT), 2) AS AVG_RIDE_COUNT
FROM hourly_sum
GROUP BY USER_TYPE
ORDER BY USER_TYPE;


--===================================================
-------------Max bike usage RT (3 Rows)
--==================================================

WITH ride_intervals AS (
    SELECT
        RIDE_ID,
        START_TIMEOFDAY_KEY,
        END_TIMEOFDAY_KEY,
        RIDEABLE_TYPE_KEY
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS
),
hourly_rides AS (
    SELECT
        r.RIDE_ID,
        u.RIDEABLE_TYPE,
        t.HOUR_24,
    FROM
        ride_intervals r
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".RIDEABLE_TYPE_DIM u
        ON r.RIDEABLE_TYPE_KEY = u.RIDEABLE_TYPE_KEY
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM t
        ON t.TIMEOFDAY_KEY BETWEEN r.START_TIMEOFDAY_KEY AND r.END_TIMEOFDAY_KEY
),
hourly_sum AS (
    SELECT RIDEABLE_TYPE,
        HOUR_24,
        COUNT(DISTINCT RIDE_ID) AS RIDE_COUNT
    FROM
        hourly_rides
    GROUP BY
        RIDEABLE_TYPE, HOUR_24
)
SELECT RIDEABLE_TYPE,
    ROUND(AVG(RIDE_COUNT), 2) AS AVG_RIDE_COUNT
FROM hourly_sum
GROUP BY RIDEABLE_TYPE
ORDER BY RIDEABLE_TYPE;


--===================================================
-------------Max bike usage ST (842 Rows)
--==================================================

WITH ride_intervals AS (
    SELECT
        RIDE_ID,
        START_TIMEOFDAY_KEY,
        END_TIMEOFDAY_KEY,
        START_STATION_KEY
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS
),
hourly_rides AS (
    SELECT
        r.RIDE_ID,
        s.STATION_NAME AS START_STATION,
        t.HOUR_24,
    FROM
        ride_intervals r
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".LOCATION_DIM s
        ON r.START_STATION_KEY = s.STATION_KEY
        JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM t
        ON t.TIMEOFDAY_KEY BETWEEN r.START_TIMEOFDAY_KEY AND r.END_TIMEOFDAY_KEY
),
hourly_sum AS (
    SELECT START_STATION,
        HOUR_24,
        COUNT(DISTINCT RIDE_ID) AS RIDE_COUNT
    FROM
        hourly_rides
    GROUP BY
        START_STATION, HOUR_24
)
SELECT START_STATION,
    ROUND(AVG(RIDE_COUNT), 2) AS AVG_RIDE_COUNT
FROM hourly_sum
GROUP BY START_STATION
ORDER BY AVG_RIDE_COUNT DESC;


--===================================================
-------------Casual-member rate-Start Station (843 Rows)
--==================================================

WITH user_type_counts AS (
    SELECT
        tf.START_STATION_KEY,
        ut.USER_TYPE,
        COUNT(tf.RIDE_ID) AS RIDE_COUNT
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.START_STATION_KEY,
        ut.USER_TYPE
)
SELECT
    DISTINCT u1.START_STATION_KEY,
    COALESCE(CAST(casual.RIDE_COUNT AS FLOAT), 0) AS CASUAL_RIDE_COUNT,
    COALESCE(member.RIDE_COUNT, 1) AS MEMBER_RIDE_COUNT,  -- Avoid division by zero
    ROUND((COALESCE(CAST(casual.RIDE_COUNT AS FLOAT), 0) / COALESCE(member.RIDE_COUNT, 1)) * 100, 2) AS CASUAL_MEMBER_RATE
FROM
    user_type_counts u1
LEFT JOIN user_type_counts casual ON u1.START_STATION_KEY = casual.START_STATION_KEY AND casual.USER_TYPE = 'casual'
LEFT JOIN user_type_counts member ON u1.START_STATION_KEY = member.START_STATION_KEY AND member.USER_TYPE = 'member'
ORDER BY
    START_STATION_KEY;



--===================================================
-------------Casual-member rate-End Station (840 Rows)
--==================================================

WITH user_type_counts AS (
    SELECT
        tf.END_STATION_KEY,
        ut.USER_TYPE,
        COUNT(tf.RIDE_ID) AS RIDE_COUNT
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.END_STATION_KEY,
        ut.USER_TYPE
)
SELECT
    DISTINCT u1.END_STATION_KEY,
    COALESCE(CAST(casual.RIDE_COUNT AS FLOAT), 0) AS CASUAL_RIDE_COUNT,
    COALESCE(member.RIDE_COUNT, 1) AS MEMBER_RIDE_COUNT,  -- Avoid division by zero
    ROUND((COALESCE(CAST(casual.RIDE_COUNT AS FLOAT), 0) / COALESCE(member.RIDE_COUNT, 1)) * 100, 2) AS CASUAL_MEMBER_RATE
FROM
    user_type_counts u1
LEFT JOIN user_type_counts casual ON u1.END_STATION_KEY = casual.END_STATION_KEY AND casual.USER_TYPE = 'casual'
LEFT JOIN user_type_counts member ON u1.END_STATION_KEY = member.END_STATION_KEY AND member.USER_TYPE = 'member'
ORDER BY
    END_STATION_KEY;



--===================================================
-------------Casual-member rate-Hour of Day (24 Rows)
--==================================================

WITH HourlyRides AS (
    SELECT
        tf.START_TIMEOFDAY_KEY,
        COUNT(CASE WHEN ut.USER_TYPE = 'casual' THEN tf.RIDE_ID END) AS CasualCount,
        COUNT(CASE WHEN ut.USER_TYPE = 'member' THEN tf.RIDE_ID END) AS MemberCount
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.START_TIMEOFDAY_KEY
),
HourlyRate AS (
    SELECT
        td.HOUR_24,
        hr.CasualCount,
        hr.MemberCount,
        CASE
            WHEN hr.CasualCount + hr.MemberCount > 0 THEN
                CAST(hr.CasualCount AS FLOAT) / (hr.CasualCount + hr.MemberCount) * 100
            ELSE 0
        END AS CasualMemberRate
    FROM
        HourlyRides hr
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".TIMEOFDAY_DIM td
        ON hr.START_TIMEOFDAY_KEY = td.TIMEOFDAY_KEY
)
SELECT
    HOUR_24,
    ROUND(AVG(CasualMemberRate), 2) AS AverageCasualMemberRate
FROM
    HourlyRate
GROUP BY
    HOUR_24
ORDER BY
    HOUR_24;


--===================================================
-------------Casual-member rate-Week Days (7 Rows)
--==================================================

WITH DailyRides AS (
    SELECT
        tf.START_DATE_KEY,
        COUNT(CASE WHEN ut.USER_TYPE = 'casual' THEN tf.RIDE_ID END) AS CasualCount,
        COUNT(CASE WHEN ut.USER_TYPE = 'member' THEN tf.RIDE_ID END) AS MemberCount
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.START_DATE_KEY
),
DailyRate AS (
    SELECT
        dd.DAY_NAME,
        hr.CasualCount,
        hr.MemberCount,
        CASE
            WHEN hr.CasualCount + hr.MemberCount > 0 THEN
                CAST(hr.CasualCount AS FLOAT) / (hr.CasualCount + hr.MemberCount) * 100
            ELSE 0
        END AS CasualMemberRate
    FROM
        DailyRides hr
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".DATE_DIM dd
        ON hr.START_DATE_KEY = dd.DATE_KEY
)
SELECT
    DAY_NAME,
    ROUND(AVG(CasualMemberRate), 2) AS AverageCasualMemberRate
FROM
    DailyRate
GROUP BY
    DAY_NAME
ORDER BY
    CASE DAY_NAME
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        ELSE 7
    END;


--===================================================
-------------Casual-member rate-Months (12 Rows)
--==================================================

WITH DailyRides AS (
    SELECT
        tf.START_DATE_KEY,
        COUNT(CASE WHEN ut.USER_TYPE = 'casual' THEN tf.RIDE_ID END) AS CasualCount,
        COUNT(CASE WHEN ut.USER_TYPE = 'member' THEN tf.RIDE_ID END) AS MemberCount
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.START_DATE_KEY
),
MonthlyRate AS (
    SELECT
        dd.MONTH_NAME,
        hr.CasualCount,
        hr.MemberCount,
        CASE
            WHEN hr.CasualCount + hr.MemberCount > 0 THEN
                CAST(hr.CasualCount AS FLOAT) / (hr.CasualCount + hr.MemberCount) * 100
            ELSE 0
        END AS CasualMemberRate
    FROM
        DailyRides hr
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".DATE_DIM dd
        ON hr.START_DATE_KEY = dd.DATE_KEY
)
SELECT
    MONTH_NAME,
    ROUND(AVG(CasualMemberRate), 2) AS AverageCasualMemberRate
FROM
    MonthlyRate
GROUP BY
    MONTH_NAME
ORDER BY
    CASE MONTH_NAME
        WHEN 'January' THEN 1
        WHEN 'February' THEN 2
        WHEN 'March' THEN 3
        WHEN 'April' THEN 4
        WHEN 'May' THEN 5
        WHEN 'June' THEN 6
        WHEN 'July' THEN 7
        WHEN 'August' THEN 8
        WHEN 'September' THEN 9
        WHEN 'October' THEN 10
        WHEN 'November' THEN 11
        ELSE 12
    END;


--===================================================
-------------Casual-member RIDEABLE_TYPE (3 Rows)
--==================================================

WITH TypeRides AS (
    SELECT
        tf.RIDEABLE_TYPE_KEY,
        COUNT(CASE WHEN ut.USER_TYPE = 'casual' THEN tf.RIDE_ID END) AS CasualCount,
        COUNT(CASE WHEN ut.USER_TYPE = 'member' THEN tf.RIDE_ID END) AS MemberCount
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.RIDEABLE_TYPE_KEY
),
RideTypeRate AS (
    SELECT
        dd.RIDEABLE_TYPE,
        hr.CasualCount,
        hr.MemberCount,
        CASE
            WHEN hr.CasualCount + hr.MemberCount > 0 THEN
                CAST(hr.CasualCount AS FLOAT) / (hr.CasualCount + hr.MemberCount) * 100
            ELSE 0
        END AS CasualMemberRate
    FROM
        TypeRides hr
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".RIDEABLE_TYPE_DIM dd
        ON hr.RIDEABLE_TYPE_KEY = dd.RIDEABLE_TYPE_KEY
)
SELECT
    RIDEABLE_TYPE,
    ROUND(AVG(CasualMemberRate), 2) AS AverageCasualMemberRate
FROM
    RideTypeRate
GROUP BY
    RIDEABLE_TYPE
ORDER BY RIDEABLE_TYPE;


--===================================================
-------------Casual-member rate-Route (100 Rows)
--==================================================

WITH StationRides AS (
    SELECT
        tf.START_STATION_KEY,
        tf.END_STATION_KEY,
        ut.USER_TYPE,
        COUNT(tf.RIDE_ID) AS RideCount
    FROM
        "PROJECT5"."PROJECT5_ANALYTICS".TRIP_FACTS tf
    JOIN "PROJECT5"."PROJECT5_ANALYTICS".USER_TYPE_DIM ut
        ON tf.USER_TYPE_KEY = ut.USER_TYPE_KEY
    GROUP BY
        tf.START_STATION_KEY,
        tf.END_STATION_KEY,
        ut.USER_TYPE
),
AggregatedRides AS (
    SELECT
        sr.START_STATION_KEY,
        sr.END_STATION_KEY,
        SUM(CASE WHEN sr.USER_TYPE = 'casual' THEN sr.RideCount ELSE 0 END) AS CasualRides,
        SUM(CASE WHEN sr.USER_TYPE = 'member' THEN sr.RideCount ELSE 0 END) AS MemberRides
    FROM
        StationRides sr
    GROUP BY
        sr.START_STATION_KEY,
        sr.END_STATION_KEY
)
SELECT
    ar.START_STATION_KEY,
    ar.END_STATION_KEY,
    -- Avoid division by zero; if no member rides, rate is set to NULL
    CASE
        WHEN ar.MemberRides > 0 THEN (CAST(ar.CasualRides AS FLOAT) / ar.MemberRides) * 100
        ELSE NULL
    END AS CasualMemberRate
FROM
    AggregatedRides ar
ORDER BY
    ar.START_STATION_KEY,
    ar.END_STATION_KEY
LIMIT 100;

SELECT *
FROM DATE_DIM
LIMIT 5;