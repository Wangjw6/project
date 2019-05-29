 async function predict(callback1,callBack2){
 	const loadedModel = await tf.loadModel('indexeddb://my-model-1');
 	var prediction=loadedModel.predict(tf.ones([1,39])).dataSync();
 	

 	var myChart = echarts.init(document.getElementById('main'));
// 绘制图表
predictHorizon = document.getElementById('f2').value;
index = Math.floor(randomFrom(0,2));
his=[[60, 62, 51, 54, 40],     [40, 42, 51, 53, 48],    [56, 52, 41, 54, 50]];
pred=[[58, 64, 53, 59, 42,35],[45, 34, 51, 57, 50, 40],[52, 50, 43, 51, 51, 50]];
var colors = ['#5793f3', '#d14a61', '#675bba'];
option = {
	backgroundColor:'rgba(128, 108, 108, 0.3)',
	tooltip: {
		trigger: 'axis'
	},
	legend: {
		textStyle: {color: 'white'},
		data:['观测值','预测值']
	},
	grid: {
		left: '3%',
		right: '4%',
		bottom: '3%',
		containLabel: true
	},
	xAxis: {
		type: 'category',
		boundaryGap: false,
	        // name:'时间',
	        data: ['6:05','6:10','6:15','6:20','6:25','6:30','6:35'],

	        axisLine:{
	        	show: true,
	        	lineStyle: {
	        		color: 'white',
	        		type: 'solid',
	        		width: 2.5
	        	}
	        }
	    },

	    yAxis: {
	    	type: 'value',
	    	axisLabel : {
	    		formatter: '{value} km/h'
	    	},
	    	axisLine:{
	    		show: true,
	    		lineStyle: {
	    			color: 'white',
	    			type: 'solid',
	    			width: 2.5
	    		}
	    	}
	    },
	    series: [
	    {
	    	name:'观测值',
	    	type:'line',
	    	itemStyle : {
	    		color:'#93DB70',
	    		width:3
	    	},
	    	data:his[index]
	    },
	    {
	    	name:'预测值',
	    	type:'line',
	    	itemStyle : {
	    		color:'red',
	    		width:3,
	    	},
	    	data:pred[index]
        // data:[prediction[0]]
    }
    ]
};

myChart.clear();
myChart.setOption(option);

    // alert(callback1);
}

function randomFrom(lowerValue,upperValue)
{
    return Math.floor(Math.random() * (upperValue - lowerValue + 1) + lowerValue);
}