stationsPayerne = function() {
    ## Read Parsivel data at Payerne.
    station10 = data.frame(number=10, name="HARAS Avenches",
        lat=46.88716, lon=7.01412, altitude=435)
    station20 = data.frame(number=20, name="Morat Airport",
        lat=46.9783217, lon=7.1299767, altitude=433)
    station30 = data.frame(number=30, name="Military Airport Payerne",
        lat=46.842456, lon=6.918397, altitude=451)
    station40 = data.frame(number=40, name="Payerne MCH Roof",
        lat=46.813323, lon=6.942843, altitude=489)
    station50 = data.frame(number=50, name="Station SwissMetNet Payerne",
        lat=46.811532, lon=6.942437, altitude=489)
    
    payerneStations = station10
    payerneStations = rbind(payerneStations, station20)
    payerneStations = rbind(payerneStations, station30)
    payerneStations = rbind(payerneStations, station40)
    payerneStations = rbind(payerneStations, station50)
    return(payerneStations)
}
