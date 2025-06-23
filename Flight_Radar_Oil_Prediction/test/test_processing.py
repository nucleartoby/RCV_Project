from FlightRadar24.api import FlightRadar24API

fr_api = FlightRadar24API()
flights = fr_api.get_flights()

military_flights = []
for flight in flights:
    if any(prefix in str(flight.callsign) for prefix in ['USAF', 'ARMY', 'NAVY', 'RCH', 'CNV']):
        military_flights.append(flight)

print(military_flights)