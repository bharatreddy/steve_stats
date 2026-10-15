import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class Events:
    event_date: datetime.datetime
    sd_start_date: datetime.datetime
    sd_end_date: datetime.datetime
    steve_start_date: datetime.datetime
    steve_end_date: Optional[datetime.datetime] = None
    radars: Optional[list] = None
    mlts: Optional[list] = None

def selected_events():

    event_dict = {}
    event_dict["event1"] = Events(
        event_date=datetime.datetime(2022,4,10),
        sd_start_date=datetime.datetime(2022,4,10,4),
        sd_end_date=datetime.datetime(2022,4,10,6),
        steve_start_date=datetime.datetime(2022,4,10,5,10),
        mlts=[21,22,23]
    )

    event_dict["event2"] = Events(
        event_date=datetime.datetime(2016,7,25),
        sd_start_date=datetime.datetime(2016,7,25,4,30),
        sd_end_date=datetime.datetime(2016,7,25,7,30),
        steve_start_date=datetime.datetime(2016,7,25,5,48),
        mlts=[20,21]
    )

    event_dict["event3"] = Events(
        event_date=datetime.datetime(2017,8,24),
        sd_start_date=datetime.datetime(2017,8,24,4),
        sd_end_date=datetime.datetime(2017,8,24,8),
        steve_start_date=datetime.datetime(2017,8,24,6,11),
        steve_end_date=datetime.datetime(2022,4,10,6,48),
        mlts=[19,20,21]
    )

    event_dict["event4"] = Events(
        event_date=datetime.datetime(2018,7,17),
        sd_start_date=datetime.datetime(2018,7,17,5),
        sd_end_date=datetime.datetime(2018,7,17,8),
        steve_start_date=datetime.datetime(2018,7,17,6),
        steve_end_date=datetime.datetime(2018,7,17,7,30),
        mlts=[20,21]
    )

    event_dict["event5"] = Events(
        event_date=datetime.datetime(2015,9,11),
        sd_start_date=datetime.datetime(2015,9,11,4,20),
        sd_end_date=datetime.datetime(2015,9,11,8),
        steve_start_date=datetime.datetime(2015,9,11,4),
        steve_end_date=datetime.datetime(2015,9,11,7),
        mlts=[20,21,22]
    )

    event_dict["event6"] = Events(
        event_date=datetime.datetime(2011,4,2),
        sd_start_date=datetime.datetime(2011,4,2,5,30),
        sd_end_date=datetime.datetime(2011,4,2,8),
        steve_start_date=datetime.datetime(2011,4,2,6,47),
        steve_end_date=datetime.datetime(2011,4,2,7,15),
        mlts=[20,21,22]
    )

    # event_dict["event7"] = Events(
    #     event_date=datetime.datetime(2011,4,20),
    #     sd_start_date=datetime.datetime(2011,4,20,7,30),
    #     sd_end_date=datetime.datetime(2011,4,20,10,30),
    #     steve_start_date=datetime.datetime(2011,4,20,8,38),
    #     steve_end_date=datetime.datetime(2011,4,20,9,10),
    #     mlts=[20,21,22]
    # )

    event_dict["event7"] = Events(
        event_date=datetime.datetime(2015,9,7),
        sd_start_date=datetime.datetime(2015,9,7,3,30),
        sd_end_date=datetime.datetime(2015,9,7,8,30),
        steve_start_date=datetime.datetime(2015,9,7,5),
        steve_end_date=datetime.datetime(2015,9,7,7),
        mlts=[20,21,22]
    )

    event_dict["event8"] = Events(
        event_date=datetime.datetime(2015,9,12),
        sd_start_date=datetime.datetime(2015,9,12,3,0),
        sd_end_date=datetime.datetime(2015,9,12,7,0),
        steve_start_date=datetime.datetime(2015,9,12,4),
        steve_end_date=datetime.datetime(2015,9,12,6),
        mlts=[20,21,22]
    )

    event_dict["event9"] = Events(
        event_date=datetime.datetime(2019,7,10),
        sd_start_date=datetime.datetime(2019,7,10,5),
        sd_end_date=datetime.datetime(2019,7,10,8),
        steve_start_date=datetime.datetime(2019,7,10,6,0),
        steve_end_date=datetime.datetime(2019,7,10,7,0),
        mlts=[20,21]
    )

    event_dict["event10"] = Events(
        event_date=datetime.datetime(2018,7,17),
        sd_start_date=datetime.datetime(2018,7,17,4,30),
        sd_end_date=datetime.datetime(2018,7,17,8,30),
        steve_start_date=datetime.datetime(2018,7,17,6,0),
        steve_end_date=datetime.datetime(2018,7,17,7,30),
        mlts=[20,21]
    )

    return event_dict