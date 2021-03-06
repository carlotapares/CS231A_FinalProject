function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function () {
        readURL(input);
    });
});

var input = document.getElementById( 'upload' );
var infoArea = document.getElementById( 'upload-label' );

function sendBase64ToServer(name, base64){
    var httpPost = new XMLHttpRequest(),
        path = "http://localhost:3002",
        data = JSON.stringify({image: base64, pose: name});
    httpPost.onreadystatechange = function(err) {
            if (httpPost.readyState == 4 && httpPost.status == 200){
                data = httpPost.response['image']
                var image = document.getElementById('image3D')
                image.src = 'data:image/png;base64,' + data
                console.log(image.src)
            } else {
                console.log(err);
            }
        };
    // Set the content type of the request to json since that's what's being sent
    httpPost.responseType = 'json';
    httpPost.timeout = 60000; // time in milliseconds
    httpPost.open("POST", path, true);
    httpPost.setRequestHeader('Content-Type', 'application/json');
    httpPost.send(data);
};

function submitImage(selText){
    var img = document.getElementById('imageResult')
    var c = document.createElement('canvas');
    c.height = img.naturalHeight;
    c.width = img.naturalWidth;
    var ctx = c.getContext('2d');

    ctx.drawImage(img, 0, 0, c.width, c.height);
    var base64String = c.toDataURL();
    sendBase64ToServer(selText, base64String);
};

$(".dropdown-menu a").click(function(){
    var selText = $(this).text();
    submitImage(selText)
  });
