elements = [];
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'convert') {
    // document.documentElement.style.filter = 'grayscale(100%)';
    // alert("heh");
  function addStyleString(str) {
      var node = document.createElement('style');
      node.innerHTML = str;
      document.body.appendChild(node);
  }
  
  addStyleString('.-sitemap-select-item-hover{outline:2px solid green!important}.-sitemap-select-item-hover,.-sitemap-select-item-hover *{background-color:rgba(0,213,0,.2)!important;background:rgba(0,213,0,.2)!important}.-sitemap-select-item-selected{outline:2px solid #c70000!important}.-sitemap-select-item-selected,.-sitemap-select-item-selected *{background-color:rgba(213,0,0,.2)!important;background:rgba(213,0,0,.2)!important}')
  
  function mouseover(event) {
      event.target.classList.add('-sitemap-select-item-hover')
  }
  function mouseout(event) {
      event.target.classList.remove('-sitemap-select-item-hover')
  }
  function clicking(event){
      element = document.elementFromPoint(event.clientX, event.clientY);
      elements.push(element);
      alert(element.innerText);

      element.style.visibility = 'hidden';
      console.log(element);
      document.removeEventListener("mouseout", mouseout);
      document.removeEventListener("mouseover", mouseover);
      document.removeEventListener("click", clicking);
  }
  
  lol = document.addEventListener('mouseover', mouseover)
  kekw = document.addEventListener('mouseout', mouseout)
  document.addEventListener('click', clicking);

  sendResponse({ message: 'Website converted to black and white.' });
  } else if (request.action === 'reset') {
    elements.forEach((element) => {
      if(element.style.visibility === "hidden")
        element.style.visibility = "visible";
      else
        element.style.visibility = "hidden";
    })
    // document.documentElement.style.filter = 'none';
    sendResponse({ message: 'Website reset to original colors.' });
  }
});