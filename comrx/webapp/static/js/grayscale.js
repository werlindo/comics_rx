(function($) {
  "use strict"; // Start of use strict

  // Smooth scrolling using jQuery easing
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      if (target.length) {
        $('html, body').animate({
          scrollTop: (target.offset().top - 70)
        }, 1000, "easeInOutExpo");
        return false;
      }
    }
  });

  
  // Closes responsive menu when a scroll trigger link is clicked
  $('.js-scroll-trigger').click(function() {
    $('.navbar-collapse').collapse('hide');
  });

  // Activate scrollspy to add active class to navbar items on scroll
  $('body').scrollspy({
    target: '#mainNav',
    offset: 100
  });

  // Collapse Navbar
  var navbarCollapse = function() {
    if ($("#mainNav").offset().top > 100) {
      $("#mainNav").addClass("navbar-shrink");
    } else {
      $("#mainNav").removeClass("navbar-shrink");
    }
  };
  // Collapse now if page is not at top
  navbarCollapse();
  // Collapse the navbar when page is scrolled
  $(window).scroll(navbarCollapse);


  // Werlindo
  
  // $('select').change(function(){
  //   var selected = $(this).find('option:selected');
  //  $('#text').html(selected.text()); 
  //  $('#value').html(selected.val()); 
  //  $('#foo').html(selected.data('foo')); 
  //  $('#url').html(selected.data('url')); 
   
  // }).change();
  
  
  
  
  $("#first-choice").change(function() {

    var $dropdown = $(this);
  
    $.getJSON("dev_files/dd_test.json", function(data) {
    
      var key = $dropdown.val();
      var vals = [];
                
      switch(key) {
        case 'beverages':
          vals = dd_test.beverages.split(",");
          break;
        case 'snacks':
          vals = dd_test.snacks.split(",");
          break;
        case 'base':
          vals = ['Please choose from above'];
      }
      
      var $secondChoice = $("#second-choice");
      $secondChoice.empty();
      $.each(vals, function(index, value) {
        $secondChoice.append("<option>" + value + "</option>");
      });
  
    });
  });

  $('#dd_comics').on('show.bs.dropdown', function () {
    // do something…
  })

})(jQuery); // End of use strict
