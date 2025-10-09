// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "hardhat/console.sol";


contract NFTTravelSwap is IERC721Receiver {
    address MyNFT = 0x146B03Bc2c0804b831640A9FFAa5b05E2c6b7dAB; //change when deployed

    event List(address indexed seller,address indexed nftAddr,uint256 indexed tokenId,uint256 price); 
    event Purchase(address indexed buyer,address indexed nftAddr,uint256 indexed tokenId,uint256 price); 
    event Revoke(address indexed seller,address indexed nftAddr,uint256 indexed tokenId); 
    event Update(address indexed seller,address indexed nftAddr,uint256 indexed tokenId,uint256 newPrice); 
    event PaymentSplit(address indexed seller, address indexed nftAddr,uint256 indexed tokenId,
        address nftCreator,uint256 sellerCut,uint256 creatorCut);

    struct Order {
        address owner;
        uint256 price;
        bool isActive;
    }
    mapping(uint256 => Order) public nftList;
    uint256 listLength;
    mapping(address => mapping(uint256 => uint256)) public creatorEarnings; //tracks the earnings of each NFT creator for each token ID.

    struct NFTMessage {
        uint256 tokenId;
        Order order;
    }
    // get all active NFTs
    function getAllActiveNFTs() public view returns (NFTMessage[] memory) {
        NFTMessage[] memory results = new NFTMessage[](listLength);
        uint256 j = 0;
        for (uint256 i = 0; i < listLength; i++) {
            if(nftList[i].isActive == true) {
                results[j] = NFTMessage(
                    i,
                    nftList[i]
                );
                j++;
            }
        }
        return results;
    }

    // get all user NFTs
    function getAllUserNFTs() public view returns (NFTMessage[] memory) {
        NFTMessage[] memory results = new NFTMessage[](listLength);
        uint256 j = 0;
        for (uint256 i = 0; i < listLength; i++) {
            if(nftList[i].owner == msg.sender) {
                results[j] = NFTMessage(
                    i,
                    nftList[i]
                );
                j++;
            }
        }
        return results;
    }


    fallback() external payable {}
    receive() external payable {}

    //put an NFT on list
    function list(uint256 _tokenId, uint256 _price) public {
        IERC721 _nft = IERC721(MyNFT); // declare IERC721
        require(_nft.getApproved(_tokenId) == address(this), "Need Approval"); // contract approval
        require(_price > 0); // price bigger than zero

        Order storage _order = nftList[_tokenId]; // define NFT owner and price
        require(_order.isActive == false, "The NFT is Active Now");
        _order.owner = msg.sender; 
        _order.price = _price;
        _order.isActive = true;
        listLength++; //track NFT list length

        // transfer NFT to contract
        _nft.safeTransferFrom(msg.sender, address(this), _tokenId);

        // emit event of list
        emit List(msg.sender, MyNFT, _tokenId, _price);
    }

    
    function purchase(uint256 _tokenId, address _nftCreator) public payable {
        uint256 _creatorCut = msg.value * 60 / 100; // crteator get 60% of total amount
        uint256 _sellerCut = msg.value - _creatorCut;

        Order storage _order = nftList[_tokenId];
        require(_order.price > 0, "Invalid Price");
        require(msg.value >= _order.price, "Increase price");
        require(_order.isActive == true, "Not Active");

        IERC721 _nft = IERC721(MyNFT);
        require(_nft.ownerOf(_tokenId) == address(this), "Invalid Order");

        _nft.safeTransferFrom(address(this), msg.sender, _tokenId); //from contract to buyer

        creatorEarnings[_nftCreator][_tokenId] += _creatorCut;
        payable(_order.owner).transfer(_sellerCut);
        payable(_nftCreator).transfer(_creatorCut);
        payable(msg.sender).transfer(msg.value - _order.price);

        _order.owner = msg.sender;
        _order.price = 0;
        _order.isActive = false;

        emit Purchase(msg.sender, MyNFT, _tokenId, _order.price);
        emit PaymentSplit(_order.owner, MyNFT, _tokenId, _nftCreator, _sellerCut, _creatorCut);
    }

    
    function revoke(uint256 _tokenId) public {
        Order storage _order = nftList[_tokenId]; 
        require(_order.owner == msg.sender, "Not Owner"); 
        require(_order.isActive == true, "Not Active");
        
        IERC721 _nft = IERC721(MyNFT);
        require(_nft.ownerOf(_tokenId) == address(this), "Invalid Order"); 
        
        _nft.safeTransferFrom(address(this), msg.sender, _tokenId); //from contract to owner

        _order.price = 0;
        _order.isActive = false;
        
        emit Revoke(msg.sender, MyNFT, _tokenId);
    }

   
    function update(uint256 _tokenId, uint256 _newPrice) public {
        require(_newPrice > 0, "Invalid Price"); 
        Order storage _order = nftList[_tokenId]; 
        require(_order.owner == msg.sender, "Not Owner"); 
        require(_order.isActive == true, "Not Active"); 
       
        IERC721 _nft = IERC721(MyNFT);
        require(_nft.ownerOf(_tokenId) == address(this), "Invalid Order"); 

        _order.price = _newPrice;
        
        emit Update(msg.sender, MyNFT, _tokenId, _newPrice);
    }

    function onERC721Received(
        address operator,
        address from,
        uint tokenId,
        bytes calldata data
    ) external override returns (bytes4) {
        return IERC721Receiver.onERC721Received.selector;
    }
}