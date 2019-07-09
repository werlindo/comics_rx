/*!
 * Start Bootstrap - Grayscale v5.0.8 (https://startbootstrap.com/template-overviews/grayscale)
 * Copyright 2013-2019 Start Bootstrap
 * Licensed under MIT (https://github.com/BlackrockDigital/startbootstrap-grayscale/blob/master/LICENSE)
 */
(function($) {
    $('select').change(function(){
        var selected = $(this).find('option:selected');
       $('#text').html(selected.text()); 
       $('#value').html(selected.val()); 
       $('#foo').html(selected.data('foo')); 
       $('#url').html(selected.data('url')); 
       $('#comic_id').html(selected.data('comic_id')); 
    }).change();
    

    $("#comic_input").change(function(){ 
        var element = $(this).find('option:selected'); 
        var myTag = element.attr("data-url"); 
        var image = document.getElementById("preview");
        image.src = myTag; 
    }); 

    $("#comic_input_2").change(function(){ 
        var element = $(this).find('option:selected'); 
        var myTag = element.attr("data-url"); 
        var image = document.getElementById("preview_2");
        image.src = myTag; 
    }); 

    $("#comic_input_3").change(function(){ 
        var element = $(this).find('option:selected'); 
        var myTag = element.attr("data-url"); 
        var image = document.getElementById("preview_3");
        image.src = myTag; 
    }); 



//     function setImage(select){
//         // var image = document.getElementById("preview")[0];
//         var image = document.getElementById("preview");
//         // image.src = select.options[select.selectedIndex].data-foo;
//         image.src = 'https://comrx.s3-us-west-2.amazonaws.com/covers/manhattan_projects.jpg'
//     } 
    
  })(jQuery); // End of use strict
