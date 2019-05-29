window.onload = function () {
			alert('请选择先验时长')
             document.getElementById('f1').addEventListener('change',function(){
                 // document.getElementById("location").innerHTML = "请选择预测时长";
                 alert('请选择预测时长')
             },false);
             document.getElementById('f2').addEventListener('change',function(){
                 // document.getElementById("location").innerHTML = "请加载训练数据";
                 alert('请加载训练数据')
             },false);
         }