for dataset in GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams SanFranciscoTraffic # MetroInterstateTrafficVolume 
do
    #python3 main.py --model KalmanHD --dataset "$dataset";
    #(( time python3 main.py --model KalmanFilter --dataset "$dataset" ) 2>&1 ) | tee KalmanFilter_"$dataset";

    d=500
    print=3000
    if [ "$dataset" = "SanFranciscoTraffic" ]; then
        learning_rate=0.001
    elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
        learning_rate=1e-05
    elif [ "$dataset" = "GuangzhouTraffic" ]; then
        learning_rate=0.001
    elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
        learning_rate=0.001
    elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
        learning_rate=0.0001
    fi

    #(( time python3 main.py --model KalmanHD --dataset "$dataset" --learning_rate "$learning_rate" --dimension_hd "$d" --print_freq "$print") 2>&1 ) | tee KalmanHD_"$dataset";

    if [ "$dataset" = "SanFranciscoTraffic" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
        learning_rate=0.00001
        hd_representation=2
        dimension_hd=1000
    elif [ "$dataset" = "GuangzhouTraffic" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
        learning_rate=0.000001
        hd_representation=4
        dimension_hd=2000
    elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
        learning_rate=0.00001
        hd_representation=1
        dimension_hd=5000
    fi

    #(( time python3 main.py --learning_rate "$learning_rate" --dimension_hd "$dimension_hd" --model RegHD --dataset "$dataset" --hd_representation "$hd_representation" ) 2>&1 ) | tee RegHD_"$dataset";

    #(( time python3 main.py --model DNN --dataset "$dataset" ) 2>&1 ) | tee DNN_"$dataset";

    (( time python3 main.py --model VAE --dataset "$dataset" ) 2>&1 ) | tee VAE_"$dataset";
done

