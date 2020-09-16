$("#message").draggable();
$("#orientation").draggable();
$("#settings").draggable();


$(".bar-dropdown").click(function() {
    var _id = $(this).parent().parent().attr('id');
    if (parseInt($(".bar-"+ _id).css("height")) > 0) {
        $(".bar-dropdown", "#"+_id).css("transform", "rotate(180deg)");
        $(".bar-"+ _id + "> .bar-wrapper").hide();
        $(".bar-"+ _id).css("height", 0);
        $("#"+_id).css("height", 36);
    } else {
        $(".bar-dropdown", "#"+_id).css("transform", "rotate(0deg)");
        if (_id == "settings") {
            $(".bar-"+ _id).css("height", 400);
            $("#"+_id).css("height", 436);
        } else {
            $(".bar-"+ _id).css("height", 200);
            $("#"+_id).css("height", 236);
        }
        $(".bar-"+ _id + "> .bar-wrapper").show();
    }
});

// master hide all elements on screen
$("#toggleGUI").click(function() {
    $(".window-item").toggle();
});

$("#showTabs").hover(
    function() {
        $("#window-tab").show();
    }, function() {
        $("#window-tab").hide();
    }
);

$("#window-tab").hover(
    function() {
        $("#window-tab").show();
    }, function() {
        $("#window-tab").hide();
    }
);

$("#offMessage").click(function() {
    if ($("#message").is(":visible")) {
        $("#message").removeClass("window-item");
        $("#message").hide();
    } else {
        $("#message").addClass("window-item");
        $("#message").show();
    }
});

$("#offOrientation").click(function() {
    if ($("#orientation").is(":visible")) {
        $("#orientation").removeClass("window-item");
        $("#orientation").hide();
    } else {
        $("#orientation").addClass("window-item");
        $("#orientation").show();
    }
});

$("#offSettings").click(function() {
    if ($("#settings").is(":visible")) {
        $("#settings").removeClass("window-item");
        $("#settings").hide();
    } else {
        $("#settings").addClass("window-item");
        $("#settings").show();
    }
});

$('#settings-box :checkbox').click(function() {
    var $this = $(this);
    // $this will contain a reference to the checkbox   
    if ($this.is(':checked')) {
        if (this.name == "wireframe") {
            wave.addWireframe();
        }
        // the checkbox was checked 
    } else {
        if (this.name == "wireframe") {
            wave.removeWireframe();
        }
        // the checkbox was unchecked
    }
});

$('#viewIso :checkbox').click(function() {
    var $this = $(this);
    if ($this.is(':checked')) {
        wave.showISO();
        $('#viewVolren :checkbox').prop('checked', false);
    } else {
        wave.showVolren();
        $('#viewVolren :checkbox').prop('checked', true); 
    }
   
});

$('#viewVolren :checkbox').click(function() {
    var $this = $(this);
    if ($this.is(':checked')) {
        wave.showVolren();
        $('#viewIso :checkbox').prop('checked', false);
    } else {
        wave.showISO();
        $('#viewIso :checkbox').prop('checked', true);
    }
});

$('#lowResolution :checkbox').click(function() {
    var $this = $(this);
    if ($this.is(':checked')) {
        wave.setRenderSizeDefault([256, 1024]);
        $('#mediumResolution :checkbox').prop('checked', false);
        $('#highResolution :checkbox').prop('checked', false);
    }
});

$('#mediumResolution :checkbox').click(function() {
    var $this = $(this);
    if ($this.is(':checked')) {
        wave.setRenderSizeDefault([512, 2048]);
        $('#lowResolution :checkbox').prop('checked', false);
        $('#highResolution :checkbox').prop('checked', false);
    }
});

$('#highResolution :checkbox').click(function() {
    var $this = $(this);
    if ($this.is(':checked')) {
        wave.setRenderSizeDefault([1024, 4096]);
        $('#lowResolution :checkbox').prop('checked', false);
        $('#mediumResolution :checkbox').prop('checked', false);
    }
});

$( "#bg-color" ).change(function() {
    wave.setBackgroundColor("#"+this.value);
});

$("#textLowerGray").change(function() {
    wave.setGrayMinValue(($("#textLowerGray").val()/255.0));
    $( "#slider-range" ).slider('values',0, ($("#textLowerGray").val()/255.0 * 100)  );
    $( "#slider-range" ).slider("refresh");
});

$("#textUpperGray").change(function() {
    wave.setGrayMaxValue(($("#textUpperGray").val()/255.0));
    $( "#slider-range" ).slider('values', 1, ($("#textUpperGray").val()/255.0 * 100)  );
    $( "#slider-range" ).slider("refresh");
});

$("#textLowerX").change(function() {
    wave.setGeometryMinX(($("#textLowerX").val()/255.0));
    $( "#slider-range-x" ).slider('values',0, ($("#textLowerX").val()/255.0 * 100)  );
    $( "#slider-range-x" ).slider("refresh");
});

$("#textUpperX").change(function() {
    wave.setGeometryMaxX(($("#textUpperX").val()/255.0));
    $( "#slider-range-x" ).slider('values', 1, ($("#textUpperX").val()/255.0 * 100)  );
    $( "#slider-range-x" ).slider("refresh");
});

$("#textLowerY").change(function() {
    wave.setGeometryMinY(($("#textLowerY").val()/255.0));
    $( "#slider-range-y" ).slider('values',0, ($("#textLowerY").val()/255.0 * 100)  );
    $( "#slider-range-y" ).slider("refresh");
});

$("#textUpperY").change(function() {
    wave.setGeometryMaxY(($("#textUpperY").val()/255.0));
    $( "#slider-range-y" ).slider('values', 1, ($("#textUpperY").val()/255.0 * 100)  );
    $( "#slider-range-y" ).slider("refresh");
});

$("#textLowerZ").change(function() {
    wave.setGeometryMinZ(($("#textLowerZ").val()/255.0));
    $( "#slider-range-z" ).slider('values',0, ($("#textLowerZ").val()/255.0 * 100)  );
    $( "#slider-range-z" ).slider("refresh");
});

$("#textUpperZ").change(function() {
    wave.setGeometryMaxZ(($("#textUpperZ").val()/255.0));
    $( "#slider-range-z" ).slider('values', 1, ($("#textUpperZ").val()/255.0 * 100)  );
    $( "#slider-range-z" ).slider("refresh");
});

$( document ).ready(function() {

    // Handler for .ready() called.
    $( "#slider-range" ).slider({
        range: true,
        min: 0,
        max: 100,
        values: [ 0, 100 ],
        slide: function( event, ui ) {
            wave.setGrayMinValue(ui.values[0]/100.0);
            wave.setGrayMaxValue(ui.values[1]/100.0);
            $("#textLowerGray").val( parseInt(ui.values[0]/100*255) );
            $("#textUpperGray").val( parseInt(ui.values[1]/100*255) );
        }
    });
    
    $( "#slider-range-x" ).slider({
        range: true,
        min: 0,
        max: 100,
        values: [ 0, 100 ],
        slide: function( event, ui ) {
            wave.setGeometryMinX(ui.values[0]/100.0)
            wave.setGeometryMaxX(ui.values[1]/100.0)
            $("#textLowerX").val( parseInt(ui.values[0]/100*255) );
            $("#textUpperX").val( parseInt(ui.values[1]/100*255) );
        }
    });
    
    $( "#slider-range-y" ).slider({
        range: true,
        min: 0,
        max: 100,
        values: [ 0, 100 ],
        slide: function( event, ui ) {
            wave.setGeometryMinY(ui.values[0]/100.0)
            wave.setGeometryMaxY(ui.values[1]/100.0)
            $("#textLowerY").val( parseInt(ui.values[0]/100*255) );
            $("#textUpperY").val( parseInt(ui.values[1]/100*255) );
        }
    });
    
    $( "#slider-range-z" ).slider({
        range: true,
        min: 0,
        max: 100,
        values: [ 0, 100 ],
        slide: function( event, ui ) {
            wave.setGeometryMinZ(ui.values[0]/100.0)
            wave.setGeometryMaxZ(ui.values[1]/100.0)
            $("#textLowerZ").val( parseInt(ui.values[0]/100*255) );
            $("#textUpperZ").val( parseInt(ui.values[1]/100*255) );
        }
    });
});
