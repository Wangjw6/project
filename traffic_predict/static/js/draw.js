// var myChart = echarts.init(document.getElementById('main'));
// // 绘制图表

// var colors = ['#5793f3', '#d14a61', '#675bba'];
// option = {
//     backgroundColor:'rgba(128, 108, 108, 0.3)',
//     tooltip: {
//         trigger: 'axis'
//     },
//     legend: {
//         textStyle: {color: 'white'},
//         data:['观测值','预测值']
//     },
//     grid: {
//         left: '3%',
//         right: '4%',
//         bottom: '3%',
//         containLabel: true
//     },
//     xAxis: {
//         type: 'category',
//         boundaryGap: false,
//         // name:'时间',
//         data: ['6:00','6:05','6:10','6:15','6:20','6:25','6:30'],

//         axisLine:{
//             show: true,
//             lineStyle: {
//                 color: 'white',
//                 type: 'solid',
//                 width: 2
//             }
//         }
//     },

//     yAxis: {
//         type: 'value',
//         axisLabel : {
//             formatter: '{value} km/h'
//         },
//         axisLine:{
//             show: true,
//             lineStyle: {
//                 color: 'white',
//                 type: 'solid',
//                 width: 2
//             }
//         }
//     }
//     // series: [
//     // {
//     //     name:'观测值',
//     //     type:'line',
//     //     itemStyle : {
//     //         color:'#93DB70',
//     //         width:3
//     //     },
//     //     data:[60, 62, 51, 54, 40]
//     // },
//     // {
//     //     name:'预测值',
//     //     type:'line',
//     //     itemStyle : {
//     //         color:'red',
//     //         width:3,
//     //     },
//     //     data:[66, 68, 51, 54, 48, 43, 41]
//     // }
//     // ]
// };


// myChart.setOption(option);