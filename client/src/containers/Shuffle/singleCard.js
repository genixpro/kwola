import React, { Component } from 'react';
import moment from 'moment';
import { SingleCardWrapper, DeleteIcon } from './shuffle.style';

export default class SingleCard extends Component {
  render() {
    const CardClass = `${this.props.view}`;
    const style = {
      zIndex: 100 - `${this.props.index !== undefined ? this.props.index : ''}`,
    };
    const { grid, list, src, title } = this.props;

    return (
      <SingleCardWrapper
        id={this.props.id}
        className={`${grid ? 'grid' : list ? 'list' : ''} ${CardClass}`}
        style={style}
      >
        <div className="cardImage">
          <img
            alt="#"
            src={`${src ? src : process.env.PUBLIC_URL + this.props.img}`}
          />
        </div>
        {/*<div className="cardContent">*/}
        {/*  <h3 className="cardTitle">{title ? title : this.props.desc}</h3>*/}
          {/*<span className="cardDate">*/}
          {/*  {moment(this.props.timestamp).format('MMM Do, YYYY')}*/}
          {/*</span>*/}
        {/*</div>*/}
        <button className="deleteBtn" onClick={this.props.clickHandler}>
          <DeleteIcon>clear</DeleteIcon>
        </button>
      </SingleCardWrapper>
    );
  }
}
