select * from pitches where season = 2024 limit 500
-----------------------
select max(season),min(season) from pitches where start_speed is null
select season,count(*) from pitches where start_speed is null group by season order by season
------------------
select pitcher ,season , count(*) from pitches
 where pitcher is not null 
 group by pitcher, season 
 order by count(*) desc, pitcher, season
 