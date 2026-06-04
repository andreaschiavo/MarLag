import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

class Calendar(object):
    def __init__(self, starting_date, duration, backward = False):
        self.starting_date = starting_date
        self.duration = duration
        
        self.end_date, self.sim_days, self.sim_time = self.calculate_simulationTime(starting_date, duration, backward)
        
        self.today = None
        self.doy = None
        self.current_year = None
        self.current_month = None

    def calculate_simulationTime(self, starting_date, duration, backward = False):
            S_DATE = date(*starting_date)
            
            SIM_YEARS, SIM_MONTHS, SIM_DAYS = duration
            
            if backward:
                if SIM_DAYS != 0 and SIM_MONTHS == 0:
                    SIM_DAYS -=1
                elif SIM_MONTHS != 0:
                    SIM_DAYS -= 1
                end_date = S_DATE-relativedelta(days = SIM_DAYS)
                end_date = end_date-relativedelta(months = SIM_MONTHS)
                end_date = end_date-relativedelta(years = SIM_YEARS)
                sim_days = (S_DATE - end_date).days
                sim_time = np.arange(sim_days, -1, -1)
                return end_date, sim_days, sim_time
            else:
                if SIM_DAYS != 0 and SIM_MONTHS == 0:
                    SIM_DAYS -=1
                elif SIM_MONTHS != 0:
                    SIM_DAYS -= 1
                end_date = S_DATE+relativedelta(days = SIM_DAYS)
                end_date = end_date+relativedelta(months = SIM_MONTHS)
                end_date = end_date+relativedelta(years = SIM_YEARS)
                sim_days = (end_date - S_DATE).days
                sim_time = np.arange(sim_days, -1, -1)
                return end_date, sim_days, sim_time
    
    # check year boundary
    def get_current_date(self, day, backward = False):
        if backward:
            current_date = self.end_date + relativedelta(days=int(day))
        else:
            current_date = self.end_date - relativedelta(days=int(day))
        return current_date
    
    def is_last_day(self, date, backward = False):
        if (backward and date.month == 1 and date.day == 1) or \
        ((not backward) and date.month == 12 and date.day == 31):
            return True
        return False

    def is_new_year(self):
        if self.today.year != self.current_year:
            self.current_year = self.today.year
            return True
        return False
    
    def is_new_month(self):
        if self.today.month != self.current_month:
            self.current_month = self.today.month
            return True
        return False
    
    def get_today(self, day, backward = False):
        today = self.end_date + relativedelta(days = int(day)) if backward else self.end_date - relativedelta(days = int(day))
        return today
    
    def get_doy(self, ):
        doy = self.today.timetuple().tm_yday
        return doy

    
def is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)