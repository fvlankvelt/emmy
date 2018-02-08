$(function() {
  $("input.nav-trigger").change(function(e) {
    if (this.checked) {
      $("body").addClass("menu-shown");
    } else {
      $("body").removeClass("menu-shown");
    }
  });
  $(window).resize(function() {
    var trigger = $("input.nav-trigger");
    trigger.prop('checked', false);
    trigger.change();
  });
});
