$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('#resultsPanel').hide();
    $('#resultCard').hide();
    $('#remedyCard').hide();
    $('#medicineCard').hide();
    $('#medicineButtonWrapper').hide();
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr( 'src', e.target.result );
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#resultCard').hide().text('');
        $('#remedyCard').hide();
        $('#medicineCard').hide();
        $('#medicineButtonWrapper').hide();
        $('#resultsPanel').hide();
        $('#regionOverlay').hide().empty();
        $('#regionNote').hide();
        $('#predictStatus').text('');
        $('#btn-predict').prop('disabled', false).find('.predict-label').text('Predict');
        readURL(this);
    });
    // Predict
    $('#btn-predict').click(function () {
        var button = $(this);
        if (button.prop('disabled')) {
            return;
        }

        var form_data = new FormData($('#upload-file')[0]);
        button.prop('disabled', true).find('.predict-label').text('Analyzing...');
        $('#predictStatus').text('Examining the leaf and calculating confidence.');
        $('#resultsPanel').hide();
        $('#regionOverlay').hide().empty();
        $('#regionNote').hide();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('#resultsPanel').fadeIn(250);
                $('#resultCard').show();
                if (data && data.primary_prediction) {
                    var readableLabel = data.primary_prediction.replace(/___/g, ' — ').replace(/_/g, ' ');
                    var text = readableLabel + ' · ' + data.confidence.toFixed(2) + '% confidence';
                    $('#resultCard').show().text(text);
                    if (data.remedy) {
                        $('#remedyCard').show();
                        $('#remedy').text(data.remedy);
                    } else {
                        $('#remedyCard').hide();
                    }
                    if (data.medicine) {
                        $('#medicineCard').show();
                        $('#medicine').text(data.medicine);
                    } else {
                        $('#medicineCard').hide();
                    }
                    if (data.medicine_button_text) {
                        var buttonUrl = data.medicine_button_url || '#';
                        $('#medicineBtn').attr('href', buttonUrl).text(data.medicine_button_text);
                        $('#medicineButtonWrapper').show();
                    } else {
                        $('#medicineButtonWrapper').hide();
                    }
                } else if (data && data.error) {
                    $('#resultCard').show().text('Error: ' + data.error);
                    $('#remedyCard').hide();
                    $('#medicineCard').hide();
                    $('#medicineButtonWrapper').hide();
                } else {
                    $('#resultCard').show().text('Result: Unable to predict');
                    $('#remedyCard').hide();
                    $('#medicineCard').hide();
                    $('#medicineButtonWrapper').hide();
                }
                var regions = data && data.suspect_regions ? data.suspect_regions : [];
                var overlay = $('#regionOverlay').empty();
                regions.forEach(function (region) {
                    $('<div></div>').css({
                        position: 'absolute',
                        left: region.x + '%',
                        top: region.y + '%',
                        width: region.width + '%',
                        height: region.height + '%',
                        border: '2px solid #ff3b30',
                        borderRadius: '4px',
                        background: 'rgba(255, 59, 48, 0.10)',
                        boxShadow: '0 0 7px rgba(255, 59, 48, 0.75)'
                    }).appendTo(overlay);
                });
                if (regions.length) {
                    overlay.show();
                    $('#regionNote').show();
                } else {
                    overlay.hide();
                    $('#regionNote').hide();
                }
                console.log('Success!', data);
            },
            error: function (xhr) {
                var message = 'Prediction failed. Please try another clear leaf image.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    message = xhr.responseJSON.error;
                }
                $('#resultsPanel').show();
                $('#resultCard').show().text(message);
                $('#remedyCard, #medicineCard, #medicineButtonWrapper').hide();
            },
            complete: function () {
                button.prop('disabled', false).find('.predict-label').text('Predict Again');
                $('#predictStatus').text('');
            }
        });
    });

});
