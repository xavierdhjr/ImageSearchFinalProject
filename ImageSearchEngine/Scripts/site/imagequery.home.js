/// <reference path="../plugins/jquery.form.min.js" />
$(function () {

    var $homeForm = $("#homeForm");
    var $response = $("#response");
    $homeForm.ajaxForm();
    $homeForm.ajaxSubmit({
        success: function (response, statusText, xhr, $form) {
                $response.text(response);
        }
    });

});